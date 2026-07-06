# SpacemiT MTMD — Multimodal Model Development Guide

> [!IMPORTANT]
> This document explains how to split a HuggingFace vision-language model (VLM)
> for the SpacemiT SMT multimodal backend and how to develop its inference
> pipeline in llama.cpp. For build/run instructions see
> [spacemit-mtmd.md](spacemit-mtmd.md).

The SpacemiT SMT multimodal solution splits a VLM into two inference subgraphs:

1. The autoregressive **language model** runs as **GGUF**, executed and sampled by
   llama.cpp / GGML.
2. The **vision encoder** runs as **ONNX**, executed by ONNX Runtime + the
   SpacemiT execution provider (EP).
3. `llama-server` receives OpenAI-compatible multimodal requests, sends the image
   into the ONNX vision backend, gets image embeddings, injects them into the LLM
   context, and continues generating text.

```text
OpenAI-compatible request
        |
        v
llama-server
        |
        +-- text prompt -> tokenizer -> llama tokens
        |
        +-- image/audio payload
              |
              v
        SMT media backend
              |
              +-- image decode / preprocess
              +-- ONNX Runtime + SpacemiT EP
              +-- image embeddings: [n_image_tokens, hidden_size]
              |
              v
        llama_decode with embd batch
        llama.cpp autoregressive generation
```

Supporting a new VLM is done in two stages:

- **Part A — Model preparation:** turn a HF VLM into a deployable
  `GGUF + ONNX + config.json` triplet. Prefer this stage; only touch the engine
  when a contract cannot be met here.
- **Part B — Engine support:** extend the llama.cpp SMT runtime when model
  preparation alone cannot satisfy the existing contract.

---

# Part A — Model preparation

Goal: produce, from a HF VLM, a deployable model directory:

- one **GGUF** text model
- one **ONNX** vision model
- one **config.json**

## A.1 Analyze the original VLM

Read the HF model files: `config.json`, `preprocessor_config.json`,
`processor_config.json`, `tokenizer_config.json`, `tokenizer.json`,
`generation_config.json`, and any custom `modeling_*.py` (for
`trust_remote_code=True`). Answer:

1. **What is the text architecture?** e.g. FastVLM ends up as a Qwen2 /
   LlavaQwen2-style text model. This decides whether `convert_hf_to_gguf.py`
   already supports the LLM.
2. **What modules are in the vision path?** vision tower, projector / resampler /
   connector, image newline / separator embedding, image token merge or spatial
   pooling.
3. **Where should ONNX be cut?** The ONNX graph should output the final image
   embeddings ready to inject into the LLM — shape equivalent to
   `[n_image_tokens, text_hidden_size]`. Do not export only the vision tower and
   leave the projector to C++, unless the engine already has that projector.
4. **What is the image preprocessing?** fixed input resolution; resize / crop /
   pad; RGB or BGR; NCHW or NHWC; dtype; rescale, mean, std.
5. **What is the prompt / image-token protocol?** the placeholder token
   (`<image>`, `<|vision_start|>`, `<|image_start|>`, or a model-private token),
   where image embeddings are inserted relative to the placeholder, and whether
   image begin/end special tokens are needed.

## A.2 Decide the split boundary

Recommended split:

```text
HF VLM
|-- text LLM                    -> GGUF
`-- vision tower + projector    -> ONNX
```

The ONNX side **should** contain the vision tower (after image resize), the
projector / connector / resampler, and every layer that maps the vision hidden
size to the LLM hidden size.

The ONNX side should **not** contain the tokenizer, chat template, autoregressive
decode, KV cache, or sampling.

Contract used by the FastVLM example:

- GGUF: `fastvlm-text-0.5B-Q4_1.gguf`
- ONNX: `fastvlm_vision.f16.onnx`
- `text_model.hidden_size = 896`
- ONNX output float array length divisible by 896

If a model's forward tightly couples image/text merge inside the LLM, record the
merge rule explicitly during preparation and let Part B decide whether the server
needs changes.

## A.3 Export the ONNX vision model

Prefer a **static** image size — it is friendlier to the SpacemiT EP and makes
warmup and profiling easier. First write a Python reference wrapper that keeps
only the image → image-embeddings path:

```python
class VisionToLlmEmbedding(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.vision_tower = hf_model.vision_tower
        self.projector = hf_model.multi_modal_projector

    def forward(self, pixel_values):
        vision_out = self.vision_tower(pixel_values)
        image_embeds = self.projector(vision_out)
        return image_embeds
```

Adjust the module names to the model source. FastVLM / LLaVA / Qwen-family models
commonly name these fields `vision_tower`, `vision_model`,
`multi_modal_projector`, `vision_projector`, `connector`, or `resampler`.

Export guidance:

- Use an opset that the ORT / SpacemiT EP supports stably.
- Use float32 NCHW input unless the SpacemiT ORT sample engine requires otherwise.
- FP16 weights are fine, but confirm the input/output contract.
- Keep the output as float — the server currently receives `std::vector<float>`.
- Prefer a fixed `1x3xHxW` input.

After export, check the input/output `name`, `dtype`, `shape`, and verify:

- Input shape is static and positive (warmup checks static input).
- Output element count is divisible by `text_model.hidden_size`.
- Output represents the final image embeddings, not logits or intermediate
  features.

## A.4 Prepare image preprocessing

FastVLM already has a default branch in the server: architecture matching
`LlavaQwen2ForCausalLM` / `llavaqwen2` / `fastvlm`, resize to `512x512`, RGB,
quantize to uint8 after resize, convert to NCHW float32, normalize to `0..1`
(`tools/mtmd/smt-vision-preprocess.cpp`).

For a new model, express fixed input size through `config.json` first:

```json
"vision_model": {
  "model_path": "vision-model.onnx",
  "input_width": 512,
  "input_height": 512
}
```

If mean/std are needed:

```json
"vision_preprocess": {
  "rescale_factor": 0.00392156862745098,
  "image_mean": [0.48145466, 0.4578275, 0.40821073],
  "image_std": [0.26862954, 0.26130258, 0.27577711]
}
```

If preprocessing cannot be expressed with existing fields, add a branch in
`smt-vision-preprocess.cpp` (Part B), or generate a preprocessed `.bin` input for
validation.

## A.5 Convert the text model to GGUF

```bash
cd /path/to/llama.cpp

python3 convert_hf_to_gguf.py /path/to/hf-model \
  --outfile /path/to/output/text-model-f16.gguf
```

Quantize if needed, e.g. to Q4_1:

```bash
./build/bin/llama-quantize \
  /path/to/output/text-model-f16.gguf \
  /path/to/output/text-model-Q4_1.gguf \
  Q4_1
```

If `convert_hf_to_gguf.py` reports the architecture is unsupported: check whether
only the text submodule can be converted; if the HF `config.json` is a multimodal
wrapper but its `text_config` is a supported architecture, make the converter
recognize the text architecture; if the text LLM is a genuinely new architecture,
go to Part B to add converter and llama.cpp architecture support.

Record: HF model name and commit, llama.cpp commit, `convert_hf_to_gguf.py` path
and any edits, GGUF dtype / quant type, tokenizer file source, and `hidden_size`.

## A.6 Generate config.json

Minimal usable format:

```json
{
  "architectures": ["LlavaQwen2ForCausalLM"],
  "vision_model": {
    "model_path": "vision-model.onnx",
    "input_width": 512,
    "input_height": 512,
    "ep_config": {
      "SPACEMIT_EP_INTRA_THREAD_NUM": "1",
      "SPACEMIT_EP_INTER_THREAD_NUM": "1",
      "SPACEMIT_EP_INTRA_THREAD_AFFINITY": "8"
    }
  },
  "text_model": {
    "model_path": "text-model-Q4_1.gguf",
    "hidden_size": 896
  }
}
```

Field notes:

- `architectures[0]` — the server uses it for model-specific behavior (image
  boundary tokens, M-RoPE, preprocessing defaults).
- `vision_model.model_path` — resolved relative to the `config.json` directory;
  absolute paths also work.
- `vision_model.input_width` / `input_height` — the ONNX fixed input size;
  recommended.
- `vision_model.ep_config` — thread and affinity config passed to the SpacemiT EP.
- `text_model.model_path` — for human/tooling record; the server still loads the
  GGUF explicitly via `-m`.
- `text_model.hidden_size` — the ONNX output embedding dimension; must be exact.

## A.7 Delivery checklist

Every new model should ship at least: `config.json`, `*.gguf`, `*.onnx`, the HF
model source and commit, the conversion/export commands, the ONNX input/output
shape record, the image-preprocessing record, the text hidden-size source, one
runnable `llama-server` command, and one end-to-end test sample. Do not ship
model files without the split description — when problems arise later, the split
description usually matters more than the file names.

---

# Part B — Engine support

Extend the llama.cpp SMT runtime only when model preparation cannot satisfy the
existing contract.

Core paths of the current SMT multimodal solution:

```text
common/arg.cpp
tools/server/server-context.cpp
tools/server/server-common.cpp
tools/server/server-smt-vision.{h,cpp}
tools/mtmd/smt-vision-wrapper.{h,cpp}
tools/mtmd/smt-vision-preprocess.{h,cpp}
tools/mtmd/CMakeLists.txt
tools/server/CMakeLists.txt
convert_hf_to_gguf.py
gguf-py/gguf/constants.py
src/llama-arch.{h,cpp}
src/llama-model*.cpp
```

## B.1 First decide whether the engine really needs changing

Prefer solving it in model preparation:

| Symptom | Prefer |
| --- | --- |
| ONNX output is not `[n_tokens, hidden_size]` | re-export ONNX with the projector/resampler included |
| image resize/normalize mismatch | fix the prep script or add a `vision_preprocess` config |
| GGUF conversion fails but the text LLM is a supported arch | make the converter pick `text_config`, don't touch the C++ runtime first |
| ORT doesn't support dynamic shape | export a static-shape ONNX |
| image token count varies but input can be fixed | fix the input size, fix the output token count |
| prompt template mismatch | fix the chat template/tokenizer, not decode |

Typical cases that **do** need engine work: `convert_hf_to_gguf.py` lacks the new
text LLM architecture; the model needs special image boundary tokens; the model
needs M-RoPE or a special position rule; existing `smt-vision-preprocess.cpp`
cannot express the preprocessing; the SpacemiT ORT sample engine's I/O API does
not match the model; the ONNX output needs reshape/transpose/slice/merge before
injection; the server's media marker rules cannot express the model's
prompt/image merge rule.

## B.2 Existing SMT server flow

Server args in `common/arg.cpp`:

- `--media-backend {auto|mtmd|smt}`
- `--vision-backend` — compatibility alias
- `--smt-config-dir DIR`

Backend selection in `tools/server/server-context.cpp`:

- `--media-backend auto` with a non-empty `--smt-config-dir` selects SMT.
- `--media-backend smt` forces SMT.
- SMT init calls `server_smt_vision_init(ctx_tgt, smt_config_dir, warmup)`.

Request parsing in `tools/server/server-common.cpp`:

- The OpenAI image payload is parsed into raw bytes.
- SMT supports `.bin` or common image formats.
- The prompt uses the internal marker `<__media__>`; the legacy `<__image__>` is
  replaced with `<__media__>`.
- If an image is present but the prompt has no marker, the server prepends one.

SMT encoding in `tools/server/server-smt-vision.cpp`:

1. `server_smt_vision_encode_media_bin`
2. image goes through `server_smt_vision_encode_image_bin`
3. `smt_vision_preprocess_if_image`
4. `smt_vision_context::encode_image`
5. check embedding shape
6. produce `server_smt_image_chunk`

SMT injection into the LLM in `server_smt_vision_decode_chunk`:

1. optionally decode an image-begin token
2. inject image embeddings via `llama_batch.embd`
3. optionally decode an image-end token
4. update `n_past`

## B.3 config.json support surface

Parsed in `tools/mtmd/smt-vision-wrapper.cpp`. Current vision-side fields:

- `architectures`
- `vision_model.model_path`
- `vision_model.input_size`
- `vision_model.input_width`
- `vision_model.input_height`
- `vision_model.ep_config`
- legacy `vision_model.spacemit_ep_intra_thread_num`
- legacy `vision_model.spacemit_ep_inter_thread_num`
- legacy `vision_model.spacemit_ep_intra_thread_affinity`
- top-level `ep_config`
- `vision_preprocess.rescale_factor`
- `vision_preprocess.image_mean`
- `vision_preprocess.image_std`
- `text_model.hidden_size`

Prefer existing fields. When adding a field, update config parsing, docs, the
example config, and the missing-field error message together.

## B.4 Add a new text LLM architecture

When `convert_hf_to_gguf.py` doesn't support the new text LLM:

1. **Find the real text architecture** in the HF config — a multimodal wrapper's
   architecture is not necessarily the text LLM. Check `text_config`,
   `language_config`, `llm_config`.
2. **Find a close model** in `convert_hf_to_gguf.py` — Qwen family →
   `Qwen2Model` / `Qwen3Model`; LLaMA family → `LlamaModel`; MoE → an existing
   MoE class.
3. **If only the wrapper architecture is new** but the text tensor names match an
   existing model: add `@ModelBase.register(...)` to the existing class, or map
   the wrapper to the text architecture in `get_model_architecture()`.
4. **If the tensor names differ:** implement `modify_tensors()` for renaming /
   transposing; check that `gguf.get_tensor_name_map()` covers all tensors.
5. **If the llama.cpp runtime lacks the architecture:** add the arch and tensor
   map in `gguf-py/gguf/constants.py`; add the arch/kv/tensor enums and names in
   `src/llama-arch.*`; add loading and graph construction in
   `src/llama-model*.cpp`.

Principles: reuse an existing architecture first; add a runtime architecture only
when the graph is genuinely different; do not write vision tensors into the text
GGUF unless the runtime explicitly needs them.

## B.5 Add or adjust vision ONNX support

Main files: `tools/mtmd/smt-vision-wrapper.{cpp,h}`.

`smt_vision_context::create()` loads `config.json`, normalizes the architecture
name, initializes the ORT API, creates the `SpineVisionModelEngine`, creates the
ONNX session, and warms up. `smt_vision_context::encode_image()` calls
`SetInputTensor`, `RunSession`, and returns `std::vector<float>`.

If the new ONNX output is not a 1-D float vector but can be reshaped into image
embeddings, do that reshape centrally in the wrapper so the server always sees:

```text
std::vector<float> embd
embd.size() == n_image_tokens * hidden_size
```

Do not let `server-common` understand a model's ONNX output details directly.

## B.6 Add image preprocessing

Main file: `tools/mtmd/smt-vision-preprocess.cpp`. Existing branches:

- Qwen3VL: `768x768`, keep the model's internal normalize, quantize to u8 after
  resize then convert to float.
- FastVLM / LlavaQwen2: `512x512`, RGB, NCHW float32, `0..1`.
- PaddleOCR: use rescale/mean/std from `vision_preprocess`.

Priority for new-model preprocessing: (1) fixed width/height →
`vision_model.input_width/input_height`; (2) standard rescale/mean/std →
`vision_preprocess`; (3) model-specific resize/crop/pad/tiling → add a C++ branch;
(4) dynamic tiles → first evaluate whether it can be fixed into a static ONNX
during preparation.

When adding preprocessing, record: input image format, resize algorithm, center
crop or pad, RGB/BGR, layout, rescale, mean/std, output dtype, output shape.

## B.7 Add image boundary token rules

Main file: `tools/server/server-smt-vision.cpp`. Functions:
`detect_image_boundary_tokens_native`, `detect_image_boundary_tokens_auto`,
`resolve_image_boundary_tokens`. Existing rules:

- Qwen2VL / Qwen2.5VL / Qwen3VL / YoutuVL: `<|vision_start|>` / `<|vision_end|>`
- Llama4: `<|image_start|>` / `<|image_end|>`
- Gemma3: `<start_of_image>` / `<end_of_image>`
- InternVL: `<img>` / `</img>`
- GLM4V: `<|begin_of_image|>` / `<|end_of_image|>`
- PaddleOCR: `<|IMAGE_START|>` / `<|IMAGE_END|>`
- LightOnOCR: `<|im_start|>` / `<|im_end|>`

FastVLM's architecture is `LlavaQwen2ForCausalLM`; with no native boundary match
it does not insert begin/end tokens — image embeddings are injected directly at
the marker position.

When adding a rule, confirm the tokenizer has the special tokens, that
`common_tokenize(... parse_special=true)` resolves them to a single token each,
and whether the boundary tokens should come from the prompt rather than be
injected by the server.

## B.8 Add M-RoPE or special position rules

Main file: `tools/server/server-smt-vision.cpp`. Functions: `arch_requires_mrope`,
`infer_image_grid_xy`, `decode_embd`. Architectures currently requiring M-RoPE:
Qwen2VL, Qwen2.5VL, Qwen3VL, GLM4V, PaddleOCR.

The M-RoPE path generates multi-dimensional positions per image embedding and
derives spatial position from `grid_nx/grid_ny`. For a new model needing special
positions, clarify: whether image tokens are a 2-D grid or 1-D sequence; how
`n_past` grows; whether text and vision positions share coordinates; whether grid
dimensions can be derived from `n_image_tokens`. Do not enable M-RoPE just because
a model is a VLM — wrong positions usually cause clear semantic drift.

## B.9 Add media marker or request-parsing rules

Main file: `tools/server/server-common.cpp`. Existing SMT prompt rules: internal
marker `<__media__>`, legacy marker `<__image__>`, auto-prepend a marker when an
image is present without one, and marker count must equal media count.

Only modify this when a new model must keep its original placeholder token and
cannot use the server injection logic. When modifying, cover: `/completion`
(`prompt_string + multimodal_data`), OpenAI chat-completions image content, file
URL, base64 data URL, multi-image, and whether history images are kept.

## B.10 Build system

SMT-related CMake: `CMakeLists.txt`, `tools/mtmd/CMakeLists.txt`,
`tools/server/CMakeLists.txt`. Build switches:

```text
LLAMA_SERVER_SMT_VISION=ON
SPACEMIT_ORT_DIR=/path/to/spacemit-ort
```

When adding source files, confirm the `tools/server` or `tools/mtmd` target
includes them, the include path finds the SpacemiT ORT headers, linking to
`libonnxruntime.so` or `libonnxruntime.a` works, and the install runtime can find
the ORT shared library.

## B.11 FastVLM current mapping

FastVLM is this project's baseline example:

- model directory: `./multimodal_llm_files`
- GGUF: `fastvlm-text-0.5B-Q4_1.gguf`
- ONNX: `fastvlm_vision.f16.onnx`
- architecture: `LlavaQwen2ForCausalLM`
- hidden size: `896`
- default preprocessing: `512x512`, RGB, NCHW float32, `0..1`
- boundary token: no native begin/end injection currently
- server: `--media-backend smt --smt-config-dir /path/to/multimodal_llm_files`

When supporting a new VLM, first make it conform to FastVLM's three-file contract,
then decide whether to extend the engine.
