#!/usr/bin/env python3
"""
Qwen3-VL / Qwen2-VL / Qwen2.5-VL 图像预处理脚本

将 jpg/png 图像转换为 llama-mtmd-cli-ep 所需的 float32 二进制文件 (.bin)。
等效于 llama.cpp 中 clip.cpp 的 clip_image_preprocess (PROJECTOR_TYPE_QWEN3VL 分支)
和 set_input (HWC -> CHW 转换)。

用法:
    python preprocess_qwen3vl.py input.jpg output.bin [--onnx-model vision.onnx]
    python preprocess_qwen3vl.py input.jpg output.bin --size 448x448
    python preprocess_qwen3vl.py input.jpg output.bin --min-pixels 3136 --max-pixels 12544

输出格式: float32, NCHW 排布 [1, 3, H, W], 直接用于 EP ONNX 视觉模型输入。
"""

import argparse
import math
import sys

import numpy as np
from PIL import Image


# ============================================================
# Qwen VL 默认归一化参数 (与 HuggingFace preprocessor_config.json 一致)
# 实际值存储在 mmproj GGUF 中，这里用标准 CLIP 值作为默认
# ============================================================

DEFAULT_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
DEFAULT_IMAGE_STD  = [0.26862954, 0.26130258, 0.27577711]

# Qwen VL 系列默认参数
DEFAULT_PATCH_SIZE  = 14
DEFAULT_MERGE_SIZE  = 2
DEFAULT_MIN_TOKENS  = 4
DEFAULT_MAX_TOKENS  = 16384


# ============================================================
# Smart Resize (等效于 clip.cpp calc_size_preserved_ratio)
# ============================================================

def smart_resize(height: int, width: int, align_size: int,
                 min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """
    保持宽高比的智能缩放，确保:
    - 输出 H 和 W 都是 align_size 的整数倍
    - min_pixels <= H*W <= max_pixels
    """
    def round_by_factor(x):
        return round(x / align_size) * align_size

    def ceil_by_factor(x):
        return math.ceil(x / align_size) * align_size

    def floor_by_factor(x):
        return math.floor(x / align_size) * align_size

    # 先按 align_size 四舍五入对齐
    h_bar = max(align_size, round_by_factor(height))
    w_bar = max(align_size, round_by_factor(width))

    # 如果超过 max_pixels，缩小
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(align_size, floor_by_factor(height / beta))
        w_bar = max(align_size, floor_by_factor(width / beta))
    # 如果小于 min_pixels，放大
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta)
        w_bar = ceil_by_factor(width * beta)

    return h_bar, w_bar


# ============================================================
# 从 ONNX 模型读取输入 shape (可选)
# ============================================================

def get_onnx_input_shape(onnx_path: str) -> tuple[int, ...] | None:
    """从 ONNX 模型的第一个输入读取静态 shape"""
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        inp = model.graph.input[0]
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                return None  # 包含动态维度
        return tuple(dims)
    except ImportError:
        print("警告: 未安装 onnx 包，无法读取模型输入 shape", file=sys.stderr)
        return None
    except Exception as e:
        print(f"警告: 读取 ONNX 模型失败: {e}", file=sys.stderr)
        return None


def get_onnx_output_token_count(onnx_path: str) -> int | None:
    """读取 ONNX 首个输出的 token 维度（若为静态）"""
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        out = model.graph.output[0]
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(d.dim_value)
            else:
                return None

        # 常见输出: [tokens, hidden] 或 [1, tokens, hidden]
        if len(dims) == 2:
            return int(dims[0])
        if len(dims) == 3 and dims[0] == 1:
            return int(dims[1])
        return None
    except Exception:
        return None


def _load_onnx_constants(onnx_path: str) -> dict[str, np.ndarray] | None:
    """读取 ONNX 中 Constant 节点和 initializer 的常量值"""
    try:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(onnx_path, load_external_data=False)
        const_map: dict[str, np.ndarray] = {}

        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value":
                    const_map[node.output[0]] = numpy_helper.to_array(attr.t)

        for init in model.graph.initializer:
            const_map[init.name] = numpy_helper.to_array(init)

        return const_map
    except Exception:
        return None


def _is_127_5_const(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=np.float32)
    return arr.size > 0 and np.allclose(arr, 127.5, atol=1e-4, rtol=0)


def onnx_has_internal_1275_norm(onnx_path: str) -> bool:
    """
    检测 ONNX 是否内置 (x - 127.5) / 127.5 归一化。
    命中后应关闭外部归一化，直接输出 0..255 的 float32 像素。
    """
    try:
        import onnx

        model = onnx.load(onnx_path, load_external_data=False)
        const_map = _load_onnx_constants(onnx_path)
        if not const_map:
            return False

        nodes = list(model.graph.node)
        check_n = min(64, len(nodes))
        for i in range(check_n):
            node = nodes[i]
            if node.op_type != "Sub":
                continue

            sub_const = None
            for inp in node.input:
                if inp in const_map and _is_127_5_const(const_map[inp]):
                    sub_const = inp
                    break
            if sub_const is None:
                continue

            sub_out = node.output[0]
            for j in range(i + 1, min(i + 12, check_n)):
                node2 = nodes[j]
                if node2.op_type != "Div":
                    continue
                if sub_out in node2.input and sub_const in node2.input:
                    return True

        return False
    except Exception:
        return False


# ============================================================
# 主处理流程
# ============================================================

def preprocess(image_path: str, output_path: str,
               target_h: int | None = None, target_w: int | None = None,
               patch_size: int = DEFAULT_PATCH_SIZE,
               merge_size: int = DEFAULT_MERGE_SIZE,
               min_tokens: int = DEFAULT_MIN_TOKENS,
               max_tokens: int = DEFAULT_MAX_TOKENS,
               image_mean: list[float] = None,
               image_std: list[float] = None,
               onnx_model: str | None = None,
               norm_mode: str = "auto"):
    """
    预处理流程:
    1. 读取图像并转为 RGB
    2. Smart Resize (保持宽高比, 对齐到 align_size)
    3. 归一化（可选）:
       - external: (pixel/255 - mean) / std
       - none:     直接使用 0..255 float32 像素（适用于 ONNX 内置归一化）
       - auto:     若检测到 ONNX 内置 (x-127.5)/127.5，则使用 none，否则 external
    4. HWC -> NCHW
    5. 写入 float32 二进制文件
    """
    if image_mean is None:
        image_mean = DEFAULT_IMAGE_MEAN
    if image_std is None:
        image_std = DEFAULT_IMAGE_STD

    # 1. 读取图像
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"原始图像大小: {orig_w}x{orig_h}")

    # 2. 确定目标尺寸
    if onnx_model:
        shape = get_onnx_input_shape(onnx_model)
        if shape is not None and len(shape) == 4:
            _, _, target_h, target_w = shape
            print(f"从 ONNX 模型读取输入 shape: {shape}")

    if target_h is not None and target_w is not None:
        new_h, new_w = target_h, target_w
        print(f"使用指定尺寸: {new_w}x{new_h}")
    else:
        align_size = patch_size * merge_size
        pixel_per_token = patch_size * patch_size * merge_size * merge_size
        min_pixels = min_tokens * pixel_per_token
        max_pixels = max_tokens * pixel_per_token
        new_h, new_w = smart_resize(orig_h, orig_w, align_size, min_pixels, max_pixels)
        print(f"Smart Resize: {new_w}x{new_h} "
              f"(align={align_size}, pixels={new_h*new_w}, "
              f"range=[{min_pixels}, {max_pixels}])")

    # 3. Resize (双线性插值, 与 clip.cpp RESIZE_ALGO_BILINEAR 一致)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    effective_norm_mode = norm_mode
    if norm_mode == "auto":
        if onnx_model and onnx_has_internal_1275_norm(onnx_model):
            effective_norm_mode = "none"
        else:
            effective_norm_mode = "external"

    # 4. 转为 numpy 并按模式处理
    pixels_u8 = np.array(img_resized, dtype=np.float32)  # [H, W, 3], 0..255
    if effective_norm_mode == "external":
        pixels_01 = pixels_u8 / 255.0
        mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
        processed = (pixels_01 - mean) / std  # [H, W, 3]
    else:
        processed = pixels_u8  # [H, W, 3], 0..255 float32

    # 5. HWC -> NCHW (等效于 clip.cpp 第 3616-3632 行的转换)
    #    clip.cpp 内存布局: [R_plane, G_plane, B_plane] 即 CHW
    #    EP ONNX 模型输入通常为 [1, 3, H, W] 即 NCHW
    chw = processed.transpose(2, 0, 1)  # [3, H, W]
    nchw = chw.reshape(1, 3, new_h, new_w)  # [1, 3, H, W]

    # 6. 写入 float32 二进制文件
    data = nchw.astype(np.float32).tobytes()
    with open(output_path, "wb") as f:
        f.write(data)

    n_tokens = None
    if onnx_model:
        n_tokens = get_onnx_output_token_count(onnx_model)
    if n_tokens is None:
        n_tokens = (new_h // patch_size // merge_size) * (new_w // patch_size // merge_size)
    print(f"输出文件: {output_path}")
    print(f"归一化模式: {effective_norm_mode}")
    print(f"数据格式: float32, shape=[1, 3, {new_h}, {new_w}]")
    print(f"文件大小: {len(data)} bytes ({len(data) // 4} floats)")
    print(f"数值范围: min={nchw.min():.6f}, max={nchw.max():.6f}")
    print(f"预计图像 token 数: {n_tokens}")


def parse_size(s: str) -> tuple[int, int]:
    """解析 WxH 或 HxW 格式的尺寸字符串"""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"无效的尺寸格式: {s}, 应为 WxH (如 448x448)")
    return int(parts[1]), int(parts[0])  # 返回 (H, W)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 图像预处理: 将 jpg/png 转为 EP ONNX 引擎所需的 .bin 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动 smart_resize (推荐)
  python preprocess_qwen3vl.py photo.jpg photo.bin

  # 指定固定尺寸 (宽x高)
  python preprocess_qwen3vl.py photo.jpg photo.bin --size 448x448

  # 从 ONNX 模型自动读取输入尺寸
  python preprocess_qwen3vl.py photo.jpg photo.bin --onnx-model vision.onnx

  # 自动检测 ONNX 内置归一化并选择处理方式（默认）
  python preprocess_qwen3vl.py photo.jpg photo.bin --onnx-model vision.onnx --norm-mode auto

  # 强制输出 0..255 float32（适用于 ONNX 已内置归一化）
  python preprocess_qwen3vl.py photo.jpg photo.bin --onnx-model vision.onnx --norm-mode none

  # 自定义 token 范围
  python preprocess_qwen3vl.py photo.jpg photo.bin --min-tokens 256 --max-tokens 1280

  # 自定义归一化参数
  python preprocess_qwen3vl.py photo.jpg photo.bin --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5
        """)

    parser.add_argument("input", help="输入图像路径 (jpg/png 等)")
    parser.add_argument("output", help="输出 .bin 文件路径")

    size_group = parser.add_argument_group("尺寸控制 (三选一)")
    size_group.add_argument("--size", type=str, default=None,
                           help="固定输出尺寸, 格式: WxH (如 448x448)")
    size_group.add_argument("--onnx-model", type=str, default=None,
                           help="ONNX 视觉模型路径, 自动从模型读取输入尺寸")

    token_group = parser.add_argument_group("Smart Resize 参数 (当未指定 --size 或 --onnx-model 时生效)")
    token_group.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                            help=f"patch 大小 (默认: {DEFAULT_PATCH_SIZE})")
    token_group.add_argument("--merge-size", type=int, default=DEFAULT_MERGE_SIZE,
                            help=f"merge 大小 (默认: {DEFAULT_MERGE_SIZE})")
    token_group.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
                            help=f"最小图像 token 数 (默认: {DEFAULT_MIN_TOKENS})")
    token_group.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                            help=f"最大图像 token 数 (默认: {DEFAULT_MAX_TOKENS})")

    norm_group = parser.add_argument_group("归一化参数")
    norm_group.add_argument("--norm-mode", choices=["auto", "external", "none"], default="auto",
                           help="归一化模式: auto(自动), external(外部归一化), none(不做外部归一化，输出0..255)")
    norm_group.add_argument("--mean", type=str, default=None,
                           help="RGB 均值, 逗号分隔 (默认: 0.48145466,0.4578275,0.40821073)")
    norm_group.add_argument("--std", type=str, default=None,
                           help="RGB 标准差, 逗号分隔 (默认: 0.26862954,0.26130258,0.27577711)")

    args = parser.parse_args()

    # 解析尺寸
    target_h, target_w = None, None
    if args.size:
        target_h, target_w = parse_size(args.size)

    # 解析归一化参数
    image_mean = [float(x) for x in args.mean.split(",")] if args.mean else None
    image_std = [float(x) for x in args.std.split(",")] if args.std else None

    preprocess(
        image_path=args.input,
        output_path=args.output,
        target_h=target_h,
        target_w=target_w,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        image_mean=image_mean,
        image_std=image_std,
        onnx_model=args.onnx_model,
        norm_mode=args.norm_mode,
    )


if __name__ == "__main__":
    main()
