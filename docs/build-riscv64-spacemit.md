> [!IMPORTANT]
> This build documentation is specific only to RISC-V SpacemiT SOCs.

## Build llama.cpp locally (for riscv64)

1. Prepare Toolchain For RISCV
~~~
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz
~~~

2. Build
Below is the build script: it requires utilizing RISC-V vector instructions for acceleration. Ensure the `GGML_CPU_RISCV64_SPACEMIT` compilation option is enabled. The currently supported optimization version is `RISCV64_SPACEMIT_IME1` and `RISCV64_SPACEMIT_IME2`, corresponding to the `RISCV64_SPACEMIT_IME_SPEC` compilation option. Compiler configurations are defined in the `riscv64-spacemit-linux-gnu-gcc.cmake` file. Please ensure you have installed the RISC-V compiler and set the environment variable via `export RISCV_ROOT_PATH={your_compiler_path}`.
```bash

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DCMAKE_TOOLCHAIN_FILE=${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
    -DCMAKE_INSTALL_PREFIX=build/installed

cmake --build build --parallel $(nproc) --config Release

pushd build
make install
popd
```

## Simulation
You can use QEMU to perform emulation on non-RISC-V architectures.

1. Download QEMU
~~~
wget https://archive.spacemit.com/spacemit-ai/qemu/jdsk-qemu-v0.0.14.tar.gz
~~~

2. Run Simulation
After build your llama.cpp, you can run the executable file via QEMU for simulation, for example:
~~~
export QEMU_ROOT_PATH={your QEMU file path}
export RISCV_ROOT_PATH_IME1={your RISC-V compiler path}

${QEMU_ROOT_PATH}/bin/qemu-riscv64 -L ${RISCV_ROOT_PATH_IME1}/sysroot -cpu max,vlen=256,elen=64,vext_spec=v1.0 ${PWD}/build/bin/llama-cli -m ${PWD}/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf -t 1
~~~
## Performance
#### Quantization Support For Matrix

* Spacemit(R) X60
~~~
model name      : Spacemit(R) X60
isa             : rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt
mmu             : sv39
uarch           : spacemit,x60
mvendorid       : 0x710
marchid         : 0x8000000058000001
~~~

Q4_0
|   Model    |   Size   | Params | backend | threads | test | t/s |
| -----------| -------- | ------ | ------- | ------- | ---- |------|
Qwen2.5 0.5B |403.20 MiB|630.17 M|   cpu   |    4    | pp512|64.12 ± 0.26|
Qwen2.5 0.5B |403.20 MiB|630.17 M|   cpu   |    4    | tg128|10.03 ± 0.01|
Qwen2.5 1.5B |1011.16 MiB| 1.78 B |   cpu   |    4    | pp512|24.16 ± 0.02|
Qwen2.5 1.5B |1011.16 MiB| 1.78 B |   cpu   |    4    | tg128|3.83 ± 0.06|
Qwen2.5 3B   | 1.86 GiB  | 3.40 B |   cpu   |    4    | pp512|12.08 ± 0.02|
Qwen2.5 3B   | 1.86 GiB  | 3.40 B |   cpu   |    4    | tg128|2.23 ± 0.02|

Q4_1
|   Model    |   Size   | Params | backend | threads | test | t/s |
| -----------| -------- | ------ | ------- | ------- | ---- |------|
Qwen2.5 0.5B |351.50 MiB|494.03 M|   cpu   |    4    | pp512|62.07 ± 0.12|
Qwen2.5 0.5B |351.50 MiB|494.03 M|   cpu   |    4    | tg128|9.91 ± 0.01|
Qwen2.5 1.5B |964.06 MiB| 1.54 B |   cpu   |    4    | pp512|22.95 ± 0.25|
Qwen2.5 1.5B |964.06 MiB| 1.54 B |   cpu   |    4    | tg128|4.01 ± 0.15|
Qwen2.5 3B   | 1.85 GiB | 3.09 B |   cpu   |    4    | pp512|11.55 ± 0.16|
Qwen2.5 3B   | 1.85 GiB | 3.09 B |   cpu   |    4    | tg128|2.25 ± 0.04|


Q4_K
|   Model    |   Size   | Params | backend | threads | test | t/s |
| -----------| -------- | ------ | ------- | ------- | ---- |------|
Qwen2.5 0.5B |462.96 MiB|630.17 M|   cpu   |    4    | pp512|9.29 ± 0.05|
Qwen2.5 0.5B |462.96 MiB|630.17 M|   cpu   |    4    | tg128|5.67 ± 0.04|
Qwen2.5 1.5B | 1.04 GiB | 1.78 B |   cpu   |    4    | pp512|10.38 ± 0.10|
Qwen2.5 1.5B | 1.04 GiB | 1.78 B |   cpu   |    4    | tg128|3.17 ± 0.08|
Qwen2.5 3B   | 1.95 GiB | 3.40 B |   cpu   |    4    | pp512|4.23 ± 0.04|
Qwen2.5 3B   | 1.95 GiB | 3.40 B |   cpu   |    4    | tg128|1.73 ± 0.00|

* Spacemit(R) A100
~~~
model name      : Spacemit(R) A100
isa             : rv64imafdcvh_zicbom_zicbop_zicboz_zicntr_zicond_zicsr_zifencei_zihintntl_zihintpause_zihpm_zimop_zaamo_zalrsc_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zcmop_zba_zbb_zbc_zbs_zkt_zvbb_zvbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkb_zvkg_zvkned_zvknha_zvknhb_zvksed_zvksh_zvkt_smaia_smstateen_ssaia_sscofpmf_sstc_svinval_svnapot_svpbmt_sdtrig
mmu             : sv39
mvendorid       : 0x710
marchid         : 0x8000000041000002
mimpid          : 0x10000000d5686200
hart isa        : rv64imafdcv_zicbom_zicbop_zicboz_zicntr_zicond_zicsr_zifencei_zihintntl_zihintpause_zihpm_zimop_zaamo_zalrsc_zawrs_zfa_zfh_zfhmin_zca_zcb_zcd_zcmop_zba_zbb_zbc_zbs_zkt_zvbb_zvbc_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkb_zvkg_zvkned_zvknha_zvknhb_zvksed_zvksh_zvkt_smaia_smstateen_ssaia_sscofpmf_sstc_svinval_svnapot_svpbmt_sdtrig
~~~

## Build and run with SMT acceleration

To enable the SpacemiT multimodal SMT extensions in addition to the default CPU build, turn on the SMT build switch and point it to a single `SPACEMIT_ORT_DIR` bundle containing:

- `include/`
- `lib/`
- `samples/`

`LLAMA_SERVER_SMT_VISION` is `OFF` by default. If it is not enabled, the build remains the native llama.cpp build without SpacemiT SMT extensions.

### Build

The SMT-enabled build needs these additional CMake definitions:

- `-DLLAMA_SERVER_SMT_VISION=ON`
- `-DSPACEMIT_ORT_DIR=<path-to-spacemit-ort-bundle>`

Example full build command:

```bash
export SPACEMIT_ORT_DIR=/path/to/spacemit-ort
export LD_LIBRARY_PATH=${SPACEMIT_ORT_DIR}/lib:${LD_LIBRARY_PATH}

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=ON \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DCMAKE_TOOLCHAIN_FILE=${PWD}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
    -DCMAKE_INSTALL_PREFIX=build/installed \
    -DLLAMA_SERVER_SMT_VISION=ON \
    -DSPACEMIT_ORT_DIR=${SPACEMIT_ORT_DIR}

cmake --build build --parallel $(nproc) --config Release

pushd build
make install
popd
```

### Run

At runtime, SMT-enabled binaries need to find the libraries under `SPACEMIT_ORT_DIR/lib`. Make sure `LD_LIBRARY_PATH` includes:

- the SMT bundle library directory
- the installed llama.cpp library directory if running from an installed package

Example:

```bash
export LD_LIBRARY_PATH=/path/to/spacemit-ort/lib:./build/installed/lib:${LD_LIBRARY_PATH}
```

For `llama-server`, also pass the SMT config directory at runtime, for example:

```bash
./build/bin/llama-server \
  -m /path/to/model.gguf \
  --vision-backend smt \
  --smt-config-dir /path/to/smt-config-dir
```

For `llama-mtmd-cli`, use the same backend selector and config directory:

```bash
./build/bin/llama-mtmd-cli \
  -m /path/to/model.gguf \
  --vision-backend smt \
  --smt-config-dir /path/to/smt-config-dir \
  --image /path/to/input.jpg \
  -p "<__media__>Describe this image."
```

Notes:

- `--smt-config-dir` should point to the directory containing `config.json` and the SMT vision ONNX model files.
- If `libspacemit_ep.so` is missing from `LD_LIBRARY_PATH`, SMT session initialization will fail at runtime.
- If `libonnxruntime.so` is missing from `LD_LIBRARY_PATH`, the SMT-enabled binary will not run successfully.
- In the current SMT integration, `config.json` only needs the fields that are actually consumed by `llama-server` / `llama-mtmd-cli`: `architectures`, `vision_model.model_path`, and `text_model.hidden_size`. A minimal example is:

```json
{
  "architectures": [
    "Qwen3VL"
  ],
  "vision_model": {
    "model_path": "./qwen3_vl_vision.onnx"
  },
  "text_model": {
    "hidden_size": 896
  }
}
```
