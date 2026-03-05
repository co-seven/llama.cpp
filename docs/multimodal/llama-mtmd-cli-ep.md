# llama-mtmd-cli-ep 使用文档

`llama-mtmd-cli-ep` 是基于 `llama.cpp` 的多模态命令行推理工具，专为 Spacemit EP ONNX 视觉引擎设计。

- **文本推理**：由 llama.cpp 加载 GGUF 格式的文本模型完成
- **视觉编码**：由 Spacemit EP ONNX 引擎完成，读取预处理的图像二进制文件（`.bin`）
- **运行模式**：支持单轮推理和交互聊天两种模式

> **状态说明（重要）**：
> 目前仅在 **FastVLM** 与 **Qwen3VL** 方案上做过端到端实测。文档中其它“架构匹配”内容仅表示代码里存在对应 token 匹配逻辑，不代表已经过完整验证。

---

## 目录

- [1. 与原生 llama-mtmd-cli 的区别](#1-与原生-llama-mtmd-cli-的区别)
- [2. 前置条件](#2-前置条件)
- [3. 命令行参数说明](#3-命令行参数说明)
- [4. 单轮推理模式](#4-单轮推理模式)
- [5. 交互聊天模式](#5-交互聊天模式)
- [6. 环境变量](#6-环境变量)
- [7. 验证范围与兼容性说明](#7-验证范围与兼容性说明)
- [8. 板端运行建议](#8-板端运行建议)
- [9. 常见报错与排查](#9-常见报错与排查)
- [10. 当前限制](#10-当前限制)

---

## 1. 与原生 llama-mtmd-cli 的区别

| 对比项 | 原生 `llama-mtmd-cli` | EP 版 `llama-mtmd-cli-ep` |
|--------|----------------------|--------------------------|
| `--mmproj` 指向 | `mmproj.gguf` 文件 | EP 配置目录（含 `config.json` + ONNX 模型） |
| `--image` 格式 | jpg / png 等常见图片格式 | 预处理后的二进制文件（`.bin`） |
| 视觉编码后端 | CLIP/GGUF | Spacemit EP ONNX 引擎 |
| 音频支持 | 支持 | 不支持（仅视觉） |

---

## 2. 前置条件

运行前需要准备以下文件：

1. **文本模型**（GGUF 格式）
   - 已验证示例：`fastvlm-text-0.5B-Q4_1.gguf`、`qwen3vl-30b-text-q4_1.gguf`

2. **EP 视觉配置目录**，目录内至少包含：
   - `config.json`：视觉模型配置，其中的 `hidden_size` **必须**与文本模型的 `n_embd` 一致
   - 视觉 ONNX 模型文件（如 `fastvlm_vision.f16.onnx`、`qwen3_vl_vision.q_replaceneg.onnx`）

3. **预处理后的图像二进制文件**
   - 例如 `input.bin`、`input2.bin`
   - 这些文件由外部预处理工具生成，不是原始的 jpg/png

> **注意**：如果 `config.json` 中的 `hidden_size` 与文本模型的 `n_embd` 不一致，程序会报错并退出。

---

## 3. 命令行参数说明

### 3.1 必需参数

| 参数 | 格式 | 说明 |
|------|------|------|
| `-m` | `-m <model.gguf>` | 文本模型文件路径（GGUF 格式）。这是 LLM 推理的核心模型。 |
| `--mmproj` (或 `-mm`) | `--mmproj <目录路径>` | EP 视觉配置目录路径，目录内需包含 `config.json` 和 ONNX 视觉模型。也可通过环境变量 `EP_CONFIG_DIR` 替代。 |

### 3.2 模式控制参数

| 参数 | 格式 | 说明 |
|------|------|------|
| `--image` | `--image <path>` | 预处理的图像二进制文件路径。支持两种方式指定多个图像：① 逗号分隔 `--image a.bin,b.bin`；② 重复参数 `--image a.bin --image b.bin`。当与 `-p` 同时提供时进入**单轮推理模式**。 |
| `-p` | `-p "<提示词>"` | 用户提示词。在提示词中使用 `<__media__>` 标记图像插入位置。当与 `--image` 同时提供时进入**单轮推理模式**。 |

> 当 `--image` 和 `-p` **同时提供**时为单轮推理模式；否则进入交互聊天模式。

### 3.3 生成控制参数

| 参数 | 格式 | 说明 |
|------|------|------|
| `-n` | `-n <数量>` | 最大生成 token 数量。默认 `-1` 表示不限制（生成到 EOS 或手动中断）。 |
| `--temp` | `--temp <温度值>` | 采样温度，控制生成随机性。值越高输出越随机，值越低越确定。 |
| `--top-k` | `--top-k <k>` | Top-K 采样参数。 |
| `--top-p` | `--top-p <p>` | Top-P（核采样）参数。 |

### 3.4 上下文与模板参数

| 参数 | 格式 | 说明 |
|------|------|------|
| `-c` | `-c <大小>` | 上下文窗口大小（token 数）。 |
| `-b` | `-b <批大小>` | 批处理大小，影响推理吞吐。 |
| `-t` | `-t <线程数>` | 推理使用的 CPU 线程数。 |
| `--chat-template` | `--chat-template <名称>` | 指定聊天模板名称。如不指定，使用模型内置模板。 |
| `--system-prompt` | `--system-prompt "<文本>"` | 系统提示词。会在对话开始前作为 system 角色消息注入。 |
| `--jinja` | `--jinja` | 启用 Jinja 模板渲染聊天格式。 |

### 3.5 关于 `<__media__>` 标记

`<__media__>` 是提示词中的图像占位标记，用于指定图像嵌入的插入位置：

- 每个 `<__media__>` 按顺序对应一个 `--image` 指定的图像文件
- `<__media__>` 的数量**必须**与图像文件数量一致
- 如果提示词中未包含 `<__media__>`，程序会自动在提示词前面为每张图像添加一个
- 旧标记 `<__image__>` 仍然兼容，会被自动替换为 `<__media__>`

---

## 4. 单轮推理模式

### 4.1 触发条件

同时提供 `--image` 和 `-p` 参数时，进入单轮推理模式。程序处理完一次请求后自动退出。

### 4.2 基本用法

```bash
llama-mtmd-cli-ep \
  -m <模型文件.gguf> \
  --mmproj <EP配置目录> \
  --image <图像.bin> \
  -p "<提示词>" \
  -n <最大token数>
```

### 4.3 单图推理示例

```bash
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files \
  --image ./multimodal_llm_files/input.bin \
  -p "<__media__>请描述这张图片的内容" \
  -n 128
```

提示词中的 `<__media__>` 会被替换为 `input.bin` 编码后的图像嵌入。如果省略 `<__media__>`，程序会自动在提示词前面插入。

### 4.4 多图推理示例

多图推理时，`<__media__>` 数量必须与图像数量一致。

**方式 A：逗号分隔图像路径**

```bash
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files \
  --image ./multimodal_llm_files/input.bin,./multimodal_llm_files/input2.bin \
  -p "<__media__>先描述第一张图片，再结合<__media__>比较第二张图片的差异" \
  -n 256
```

**方式 B：重复 `--image` 参数**

```bash
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files \
  --image ./multimodal_llm_files/input.bin \
  --image ./multimodal_llm_files/input2.bin \
  -p "<__media__>先描述第一张图片，再结合<__media__>比较第二张图片的差异" \
  -n 256
```

### 4.5 带系统提示词的推理

```bash
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files \
  --image ./multimodal_llm_files/input.bin \
  --system-prompt "你是一个专业的图像分析助手，请用中文详细描述图片内容。" \
  -p "<__media__>请分析这张图" \
  -n 256
```

### 4.6 处理流程

1. 加载文本模型和 EP 视觉引擎
2. 如果有系统提示词，先评估系统消息
3. 检查提示词中是否包含 `<__media__>`，如果没有则自动在前面为每张图片添加
4. 将提示词按 `<__media__>` 分割为文本块和图像块
5. 依次处理每个块：
   - **文本块**：tokenize 后送入 LLM 解码
   - **图像块**：通过 EP 视觉引擎编码为嵌入向量，注入图像边界 token 后送入 LLM
6. 采样生成响应，直到遇到 EOS token 或达到最大 token 数
7. 输出响应并退出

---

## 5. 交互聊天模式

### 5.1 触发条件

当 `--image` 和 `-p` **未同时提供**时，进入交互聊天模式。程序显示提示符 `>` 等待用户输入。

### 5.2 启动命令

```bash
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files
```

启动后会显示：

```
Running in chat mode (EP vision), available commands:
  /image <path>    load a preprocessed image binary
  /clear           clear the chat history
  /quit or /exit   exit the program
```

### 5.3 聊天命令

| 命令 | 说明 |
|------|------|
| `/image <path>` | 加载一个预处理的图像二进制文件，并在当前输入中追加一个 `<__media__>` 标记。可以在发送文本前多次使用来加载多张图片。 |
| `/clear` | 清空聊天历史和上下文，恢复到初始状态。如有系统提示词会重新评估。 |
| `/quit` 或 `/exit` | 退出程序。 |
| `Ctrl+C` | 中断当前生成。再按一次 `Ctrl+C` 退出程序。 |
| 其他文本 | 作为用户消息发送给模型，触发模型生成响应。 |

### 5.4 交互示例

#### 纯文本聊天

```
> 你好，请介绍一下你自己
你好！我是一个多模态AI助手，可以理解文本和图像...

> 1+1等于几？
1+1等于2。
```

#### 图文聊天（先加载图片，再提问）

```
> /image ./multimodal_llm_files/input.bin
./multimodal_llm_files/input.bin image binary loaded

> 请描述这张图片的内容
这张图片展示了...

> 图片中有几个人？
图片中有...
```

#### 多图聊天

```
> /image ./multimodal_llm_files/input.bin
./multimodal_llm_files/input.bin image binary loaded

> /image ./multimodal_llm_files/input2.bin
./multimodal_llm_files/input2.bin image binary loaded

> 比较这两张图片的区别
第一张图片...第二张图片...

> /clear
Chat history cleared
```

### 5.5 聊天模式工作流程

1. 用户输入 `/image <path>`：图像路径加入待处理队列，`<__media__>` 标记追加到当前消息内容
2. 用户输入普通文本：文本追加到当前消息内容，触发推理
3. 推理时将消息通过聊天模板格式化，分块处理文本和图像
4. 生成的响应存入聊天历史，支持多轮上下文
5. `/clear` 会清空聊天历史和 KV Cache，重新开始对话

---

## 6. 环境变量

### 6.1 `EP_CONFIG_DIR`

当未通过 `--mmproj` 指定 EP 配置目录时，程序会尝试从此环境变量读取。

```bash
export EP_CONFIG_DIR=./multimodal_llm_files
llama-mtmd-cli-ep -m text.gguf --image input.bin -p "描述图片"
```

### 6.2 `MTMD_EP_IMAGE_BOUNDARY`

控制图像边界 token 的注入策略。图像边界 token（如 `<|vision_start|>` / `<|vision_end|>`）用于告诉 LLM 图像嵌入的起止位置。

| 值 | 说明 |
|----|------|
| `native`（默认，推荐） | 根据模型架构名称（如 qwen2vl、gemma3 等）自动匹配对应的边界 token |
| `auto` 或 `detect` | 不依赖架构名称，尝试从词汇表中探测已知的边界 token 对 |
| `none`、`off` 或 `0` | 不注入任何图像边界 token |

```bash
export MTMD_EP_IMAGE_BOUNDARY=native
```

### 6.3 `MTMD_EP_MEDIA_ANCHOR`

控制多图提示词的锚点规范化。启用后会为多图提示词自动添加图像编号标注，帮助模型区分多张图片。

| 值 | 说明 |
|----|------|
| `off`（默认） | 不进行规范化 |
| `auto` | 仅对已知支持的架构（如 LLaVA-Qwen2）应用规范化 |
| `on`、`1` 或 `true` | 对所有多图提示词强制应用规范化 |

规范化效果示例：

```
# 原始提示词：
<__media__>先描述第一张，再结合<__media__>比较第二张差异

# 规范化后：
Please align images by order index. Image #1 maps to the first media slot, Image #2 to the second, and so on.
[Image 1 Begin]
<__media__>
[Image 1 End]
先描述第一张，再结合
[Image 2 Begin]
<__media__>
[Image 2 End]
比较第二张差异
```

---

## 7. 验证范围与兼容性说明

### 7.1 已完成端到端验证的模型

当前仅验证了以下两类方案：

| 方案 | `config.json` 中常见 `architectures` | 状态 |
|------|--------------------------------------|------|
| FastVLM | `LlavaQwen2ForCausalLM` | 已验证 |
| Qwen3VL | `Qwen3VL` | 已验证 |

### 7.2 图像边界 token 匹配规则（代码层）

EP 版工具会根据文本模型架构名称匹配图像边界 token。下表表示**代码中存在匹配逻辑**，不等同于“已完成端到端验证”：

| 模型架构 | 图像开始 token | 图像结束 token |
|---------|---------------|---------------|
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL / YoutuVL | `<\|vision_start\|>` | `<\|vision_end\|>` |
| Llama4 | `<\|image_start\|>` | `<\|image_end\|>` |
| Gemma3 | `<start_of_image>` | `<end_of_image>` |
| InternVL | `<img>` | `</img>` |
| GLM-4V | `<\|begin_of_image\|>` | `<\|end_of_image\|>` |
| PaddleOCR | `<\|IMAGE_START\|>` | `<\|IMAGE_END\|>` |
| LightonOCR | `<\|im_start\|>` | `<\|im_end\|>` |

如果你的模型架构不在上面列表中，可以尝试设置 `MTMD_EP_IMAGE_BOUNDARY=auto` 让程序自动探测，或设置为 `none` 跳过边界 token 注入。

> 对于未验证模型，建议先做小样本功能验证（单图、短上下文、固定提示词），再进行批量测试。

---

## 8. 板端运行建议

### 8.1 设置动态库路径

```bash
export LD_LIBRARY_PATH=./llamainstall/lib/:ort_installed/lib/:installed/lib/
```

### 8.2 记录运行日志

建议将输出保存到日志文件，方便调试：

```bash
LOG_FILE=./ep_run_$(date +%Y%m%d_%H%M%S).log
llama-mtmd-cli-ep \
  -m ./multimodal_llm_files/fastvlm-text-0.5B-Q4_1.gguf \
  --mmproj ./multimodal_llm_files \
  --image ./multimodal_llm_files/input.bin \
  -p "<__media__>请描述这张图" \
  -n 128 \
  2>&1 | tee "$LOG_FILE"
```

### 8.3 推理性能调优

- `-t <线程数>`：根据板端 CPU 核心数设置合理的线程数
- `-b <批大小>`：增大批大小可以提高吞吐，但会增加内存占用
- `-c <上下文大小>`：根据需要调整上下文窗口大小，减小可以节省内存

---

## 9. 常见报错与排查

### `Missing EP config directory`

**原因**：未通过 `--mmproj` 指定 EP 配置目录，且环境变量 `EP_CONFIG_DIR` 也未设置。

**解决**：添加 `--mmproj <目录路径>` 或设置 `export EP_CONFIG_DIR=<目录路径>`。

### `Number of images (...) does not match number of media markers (...)`

**原因**：提示词中 `<__media__>` 标记的数量与 `--image` 指定的图像文件数量不一致。

**解决**：确保每张图片对应一个 `<__media__>` 标记。例如两张图片需要两个 `<__media__>`。

### `model n_embd (...) != EP hidden_size (...)`

**原因**：文本模型的嵌入维度与 EP 配置中的 `hidden_size` 不匹配。

**解决**：使用同一套模型导出的文本 GGUF 和 EP 视觉配置。

### `Invalid image embedding shape from EP`

**原因**：EP 视觉引擎对图像的编码输出形状不正确。可能是 `.bin` 文件格式或来源不正确，或视觉 ONNX 模型文件有问题。

**解决**：检查 `.bin` 文件是否由正确的预处理工具生成，且与当前的 ONNX 视觉模型匹配。

### `Failed to decode token` / `Failed to decode text chunk`

**原因**：LLM 解码过程中出错，可能是上下文窗口溢出或内存不足。

**解决**：尝试增大 `-c` 上下文大小，或减少输入图片数量。

---

## 10. 当前限制

- 仅支持视觉输入（图像），**不支持音频输入**
- 图像输入必须是预处理后的二进制文件（`.bin`），不能直接使用 jpg/png
- 端到端仅在 FastVLM、Qwen3VL 上实测通过；其它模型请视为“未验证状态”
- 建议使用默认的环境变量配置以保持稳定：
  - `MTMD_EP_IMAGE_BOUNDARY=native`
  - `MTMD_EP_MEDIA_ANCHOR=off`
