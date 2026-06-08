#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
    _mistral_common_installed,
    _mistral_import_error_msg,
)


LINGBOT_MAP_ARCH = "lingbot-map"
LINGBOT_MAP_DEFAULT_CHECKPOINT = Path("/home/cailinxi/modelzoo/lingbot-map/hf_model/lingbot-map.pt")
LINGBOT_MAP_DEFAULT_OUTFILE = Path("/home/cailinxi/modelzoo/lingbot-map/mtmd_model/lingbot-map-agg-camera-f32.gguf")


def unwrap_lingbot_map_state_dict(obj: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint must contain a state dict, got {type(obj)!r}")

    state = OrderedDict()
    for key, value in obj.items():
        if isinstance(value, torch.Tensor):
            state[str(key)] = value.detach().cpu()
    if not state:
        raise ValueError("checkpoint does not contain tensor entries")
    return state


def lingbot_map_selected_tensor_names(
        state: OrderedDict[str, torch.Tensor],
        include_patch_embed: bool,
        include_depth_head: bool) -> list[str]:
    names: list[str] = []
    for name in state:
        if name.startswith("aggregator."):
            if not include_patch_embed and name.startswith("aggregator.patch_embed."):
                continue
            names.append(name)
        elif name.startswith("camera_head."):
            names.append(name)
        elif include_depth_head and name.startswith("depth_head."):
            names.append(name)
    return names


def lingbot_map_count_indexed_modules(names: Iterable[str], prefix: str) -> int:
    indices: set[int] = set()
    needle = prefix + "."
    for name in names:
        if not name.startswith(needle):
            continue
        rest = name[len(needle):]
        first = rest.split(".", 1)[0]
        if first.isdigit():
            indices.add(int(first))
    return max(indices) + 1 if indices else 0


def lingbot_map_infer_metadata(
        state: OrderedDict[str, torch.Tensor],
        selected: list[str],
        include_patch_embed: bool,
        include_depth_head: bool) -> dict[str, object]:
    camera_token = state.get("aggregator.camera_token")
    if camera_token is None:
        raise KeyError("missing required tensor: aggregator.camera_token")

    embed_dim = int(camera_token.shape[-1])
    num_camera_token_variants = int(camera_token.shape[1])
    num_register_tokens = int(state["aggregator.register_token"].shape[2]) if "aggregator.register_token" in state else 0
    has_scale_token = "aggregator.scale_token" in state
    num_special_tokens = 1 + num_register_tokens + (1 if has_scale_token else 0)
    frame_blocks = lingbot_map_count_indexed_modules(selected, "aggregator.frame_blocks")
    global_blocks = lingbot_map_count_indexed_modules(selected, "aggregator.global_blocks")
    camera_blocks = lingbot_map_count_indexed_modules(selected, "camera_head.trunk")

    patch_proj = state.get("aggregator.patch_embed.patch_embed.proj.weight")
    patch_size = int(patch_proj.shape[-1]) if patch_proj is not None else 14

    camera_qkv = state.get("camera_head.trunk.0.attn.qkv.weight")
    camera_dim = int(camera_qkv.shape[1]) if camera_qkv is not None else embed_dim * 2
    camera_pose_dim = int(state["camera_head.empty_pose_tokens"].shape[-1]) if "camera_head.empty_pose_tokens" in state else 9

    return {
        "schema_version": 1,
        "component": "aggregator_camera_head",
        "includes_patch_embed": bool(include_patch_embed),
        "includes_depth_head": bool(include_depth_head),
        "embed_dim": embed_dim,
        "camera_dim": camera_dim,
        "camera_pose_dim": camera_pose_dim,
        "patch_size": patch_size,
        "num_register_tokens": num_register_tokens,
        "num_special_tokens": num_special_tokens,
        "num_camera_token_variants": num_camera_token_variants,
        "has_scale_token": has_scale_token,
        "aggregator_frame_block_count": frame_blocks,
        "aggregator_global_block_count": global_blocks,
        "camera_trunk_block_count": camera_blocks,
        "aa_order": ["frame", "global"],
        "aa_block_size": 1,
        "rope_freq": 100.0,
        "resnet_mean": [0.485, 0.456, 0.406],
        "resnet_std": [0.229, 0.224, 0.225],
    }


def lingbot_map_add_metadata(writer: gguf.GGUFWriter, meta: dict[str, object], outtype: str) -> None:
    writer.add_name("LingBot-MAP aggregator + camera head")
    writer.add_type("model")
    writer.add_description("LingBot-MAP non-LLM GGUF containing aggregator and camera head tensors.")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16 if outtype == "f16" else gguf.LlamaFileType.ALL_F32))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    for key, value in meta.items():
        full_key = f"{LINGBOT_MAP_ARCH}.{key}"
        if isinstance(value, bool):
            writer.add_bool(full_key, value)
        elif isinstance(value, int):
            writer.add_uint32(full_key, value)
        elif isinstance(value, float):
            writer.add_float32(full_key, value)
        elif isinstance(value, str):
            writer.add_string(full_key, value)
        elif isinstance(value, list):
            writer.add_array(full_key, value)
        else:
            raise TypeError(f"unsupported metadata value for {key}: {type(value)!r}")


def lingbot_map_tensor_to_numpy(tensor: torch.Tensor, outtype: str) -> np.ndarray:
    if tensor.dtype.is_floating_point:
        if outtype == "f16":
            return tensor.to(torch.float16).numpy()
        return tensor.to(torch.float32).numpy()

    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return tensor.numpy()

    raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")


def write_lingbot_map_gguf(args: argparse.Namespace) -> None:
    outtype = "f32" if args.outtype == "auto" else args.outtype
    if outtype not in ("f32", "f16"):
        raise ValueError("LingBot-MAP GGUF conversion only supports --outtype f32 or f16")

    checkpoint_path = args.checkpoint or LINGBOT_MAP_DEFAULT_CHECKPOINT
    outfile = args.outfile or LINGBOT_MAP_DEFAULT_OUTFILE

    logger.info("Loading LingBot-MAP checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = unwrap_lingbot_map_state_dict(checkpoint)
    selected = lingbot_map_selected_tensor_names(state, args.include_patch_embed, args.include_depth_head)
    if not selected:
        raise ValueError("no LingBot-MAP tensors selected for conversion")

    meta = lingbot_map_infer_metadata(state, selected, args.include_patch_embed, args.include_depth_head)
    total_params = sum(state[name].numel() for name in selected)
    total_bytes = sum(lingbot_map_tensor_to_numpy(state[name], outtype).nbytes for name in selected)

    logger.info("Selected LingBot-MAP tensors: %d", len(selected))
    logger.info("Selected LingBot-MAP parameters: %.3f M", total_params / 1e6)
    logger.info("Selected LingBot-MAP tensor bytes: %.3f MiB", total_bytes / (1024 * 1024))
    for key, value in meta.items():
        logger.info("LingBot-MAP meta %s = %s", key, value)

    if args.dry_run:
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(outfile, LINGBOT_MAP_ARCH)
    lingbot_map_add_metadata(writer, meta, outtype)

    for name in selected:
        arr = lingbot_map_tensor_to_numpy(state[name].contiguous(), outtype)
        writer.add_tensor(name, arr)

    logger.info("Writing LingBot-MAP GGUF: %s", outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("LingBot-MAP GGUF conversion done")


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G")

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a huggingface model to a GGML compatible file")
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="extract only the vocab",
    )
    parser.add_argument(
        "--outfile", type=Path,
        help="path to write to; default: based on input. {ftype} will be replaced by the outtype.",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="auto",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, tq1_0 or tq2_0 for ternary, and auto for the highest-fidelity 16-bit float type",
    )
    parser.add_argument(
        "--lingbot-map", action="store_true",
        help="Export LingBot-MAP aggregator and camera_head tensors from a PyTorch checkpoint to GGUF.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Path to LingBot-MAP .pt checkpoint. Only used with --lingbot-map.",
    )
    parser.add_argument(
        "--include-patch-embed", action="store_true",
        help="Also include aggregator.patch_embed.* tensors when converting LingBot-MAP.",
    )
    parser.add_argument(
        "--include-depth-head", action="store_true",
        help="Also include depth_head.* tensors when converting LingBot-MAP.",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=str,
        help="directory containing model file or huggingface repository ID (if --remote)",
        nargs="?",
    )
    parser.add_argument(
        "--use-temp-file", action="store_true",
        help="use the tempfile library while processing (helpful when running out of memory, process killed)",
    )
    parser.add_argument(
        "--no-lazy", action="store_true",
        help="use more RAM by computing all outputs before writing (use in case lazy evaluation is broken)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="name of the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )
    parser.add_argument(
        "--split-max-tensors", type=int, default=0,
        help="max tensors in each split",
    )
    parser.add_argument(
        "--split-max-size", type=str, default="0",
        help="max size per split N(M|G)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="only print out a split plan and exit, without writing any new files. In --lingbot-map mode, print selected tensors and inferred metadata without writing GGUF.",
    )
    parser.add_argument(
        "--no-tensor-first-split", action="store_true",
        help="do not add tensors to the first split (disabled by default)"
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Specify the path for an authorship metadata override file"
    )
    parser.add_argument(
        "--print-supported-models", action="store_true",
        help="Print the supported models"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="(Experimental) Read safetensors file remotely without downloading to disk. Config and tokenizer files will still be downloaded. To use this feature, you need to specify Hugging Face model repo name instead of a local directory. For example: 'HuggingFaceTB/SmolLM2-1.7B-Instruct'. Note: To access gated repo, set HF_TOKEN environment variable to your Hugging Face token.",
    )
    parser.add_argument(
        "--mmproj", action="store_true",
        help="Export multimodal projector (mmproj) for vision models. This will only work on some vision models. An 'mmproj-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--mtp", action="store_true",
        help="Export only the multi-token prediction (MTP) head as a separate GGUF, suitable for use as a speculative draft. An 'mtp-' prefix will be added to the output file name.",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="Exclude the multi-token prediction (MTP) head from the converted GGUF. Pair with --mtp on a second run to publish trunk and MTP as two files. Note: the split form duplicates embeddings, but even though the bundled default is more space-efficient overall, this allows differing quantization which may be more performant.",
    )
    parser.add_argument(
        "--mistral-format", action="store_true",
        help="Whether the model is stored following the Mistral format.",
    )
    parser.add_argument(
        "--disable-mistral-community-chat-template", action="store_true",
        help=(
            "Whether to disable usage of Mistral community chat templates. If set, use the Mistral official `mistral-common` library for tokenization and detokenization of Mistral models. "
            "Using `mistral-common` ensure correctness and zero-day support of tokenization for models converted from the Mistral format but requires to manually setup the tokenization server."
        )
    )

    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help=("Whether to include sentence-transformers dense modules. "
              "It can be used for sentence-transformers models, like google/embeddinggemma-300m. "
              "Default these modules are not included.")
    )

    parser.add_argument(
        "--fuse-gate-up-exps", action="store_true",
        help="Fuse gate_exps and up_exps tensors into a single gate_up_exps tensor for MoE models.",
    )
    parser.add_argument(
        "--fp8-as-q8", action="store_true",
        help="Store tensors dequantized from FP8 as Q8_0 instead of BF16/F16.",
    )

    args = parser.parse_args()
    if not args.print_supported_models and not args.lingbot_map and args.model is None:
        parser.error("the following arguments are required: model")
    return args


def main() -> None:
    args = parse_args()

    if args.print_supported_models:
        logger.error("Supported models:")
        print_registered_models()
        sys.exit(0)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.lingbot_map:
        write_lingbot_map_gguf(args)
        return

    if args.remote:
        hf_repo_id = args.model
        from huggingface_hub import snapshot_download
        allowed_patterns = ["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model"]
        if args.sentence_transformers_dense_modules:
            # include sentence-transformers dense modules safetensors files
            allowed_patterns.append("*.safetensors")
        local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)
        dir_model = Path(local_dir)
        logger.info(f"Downloaded config and tokenizer to {local_dir}")
    else:
        hf_repo_id = None
        dir_model = Path(args.model)

    if not dir_model.is_dir():
        logger.error(f'Error: {dir_model} is not a directory')
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = args.split_max_tensors > 0 or args.split_max_size != "0"
    if args.use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if args.outfile is not None:
        fname_out = args.outfile
    elif hf_repo_id:
        # if remote, use the model ID as the output file name
        fname_out = Path("./" + hf_repo_id.replace("/", "-") + "-{ftype}.gguf")
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    is_mistral_format = args.mistral_format
    if is_mistral_format and not _mistral_common_installed:
        raise ImportError(_mistral_import_error_msg)
    disable_mistral_community_chat_template = args.disable_mistral_community_chat_template

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_type = ModelType.MMPROJ if args.mmproj else ModelType.TEXT
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        if not is_mistral_format:
            model_architecture = get_model_architecture(hparams, model_type)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                model_class = get_model_class(model_architecture, mmproj=(model_type == ModelType.MMPROJ))
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported")
                sys.exit(1)
        elif args.mmproj:
            assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
            from conversion.pixtral import PixtralModel
            model_class = PixtralModel
        elif "moe" in hparams:
            from conversion.mistral import MistralMoeModel
            model_class = MistralMoeModel
        else:
            from conversion.mistral import MistralModel
            model_class = MistralModel

        if args.mtp and args.no_mtp:
            logger.error("--mtp and --no-mtp are mutually exclusive")
            sys.exit(1)

        if args.mtp or args.no_mtp:
            from conversion.qwen import _Qwen35MtpMixin
            if not issubclass(model_class, _Qwen35MtpMixin):
                logger.error("--mtp / --no-mtp are only supported for Qwen3.5/3.6 text variants today")
                sys.exit(1)
            if args.no_mtp:
                model_class.no_mtp = True
            if args.mtp:
                model_class.mtp_only = True

        model_instance = model_class(dir_model, output_type, fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split,
                                     remote_hf_model_id=hf_repo_id, disable_mistral_community_chat_template=disable_mistral_community_chat_template,
                                     sentence_transformers_dense_modules=args.sentence_transformers_dense_modules,
                                     fuse_gate_up_exps=args.fuse_gate_up_exps,
                                     fp8_as_q8=args.fp8_as_q8,
                                     )

        if args.vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            logger.info(f"Model vocab successfully exported to {model_instance.fname_out}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")


if __name__ == '__main__':
    main()
