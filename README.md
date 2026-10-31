# MossInfer

A native C++/CUDA inference engine for the MOSS-Image text-to-image model (5B DiT, Qwen3.5 text encoder, FLUX.2 VAE), with no PyTorch or Python at run time. It plans memory against the GPU it finds and streams the model when it does not fit.

Models: MOSS-Image Base and SFT only (`OpenMOSS-Team/MOSS-Image-Base`, `OpenMOSS-Team/MOSS-Image-SFT`); other models are refused ([supported models](docs/integration.md#supported-models)).

Version 2.1.0. Supported: Linux x86-64 and Windows x64 (MSVC), each with a CUDA 12 and a CUDA 13 build, as a full package and a `-system-cuda` package. Turing (compute capability 7.5, e.g. RTX 2080) is supported by builds configured with `-DMOSSINFER_SM75=ON`: fp16 tensor-core GEMMs, INT8/INT4 ConvRot files (INT4 through the INT8 unpack fallback), exact attention and the VAE then run on sm_75 kernels, and an unset attention resolves to `exact`. The full support matrix is in [docs/integration.md](docs/integration.md#support-status).

The caller picks the quality tier: `reference` (bf16, exact), `high` (the loaded DiT's GEMMs with 8-bit attention) or `turbo` (a 4-bit DiT). An INT8 DiT with `high` is the everyday choice: 2.5 times the bf16 speed on an RTX 5090, in 40 % less VRAM ([performance](docs/integration.md#performance)). Upgrading from 1.x: [migration](docs/integration.md#migrating-from-1x).

```sh
moss-infer generate --model MOSS-Image/sft --dit moss_image_sft_int8_convrot.safetensors --tier high \
  --prompt "a red apple on a wooden table" -o apple.png
```

- [docs/integration.md](docs/integration.md): building the engine into an application. Start here.
- [docs/c-api.md](docs/c-api.md): the C library, `include/mossinfer/mossinfer.h`.
- [docs/server-protocol.md](docs/server-protocol.md): `moss-server`, the JSON-lines sidecar process.

Build (CUDA 12.x or 13.x, CMake ≥ 3.28, Ninja; Linux: GCC ≥ 12 and GoogleTest; Windows: a Visual Studio 2022 x64 developer prompt):

```sh
cmake --preset linux-cuda13 && cmake --build --preset linux-cuda13
CUDA_VISIBLE_DEVICES="" ctest --preset linux-cuda13 -L '^cpu$'
```

The other presets are `linux-cuda12`, `windows-msvc-cuda13` and `windows-msvc-cuda12` ([docs/integration.md](docs/integration.md#getting-the-binaries)). Add `-DMOSSINFER_SM75=ON` to also build the Turing (sm_75) kernels and accept compute capability 7.5 devices; without it the engine is Ampere-or-newer as before.

Licensed under Apache-2.0 (`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).
