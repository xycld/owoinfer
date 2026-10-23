# MossInfer

A native C++/CUDA inference engine for the MOSS-Image text-to-image model (5B DiT, Qwen3.5 text encoder, FLUX.2 VAE), with no PyTorch or Python at run time. It plans memory against the GPU it finds and streams the model when it does not fit.

Models: MOSS-Image Base and SFT only (`OpenMOSS-Team/MOSS-Image-Base`, `OpenMOSS-Team/MOSS-Image-SFT`); other models are refused ([supported models](docs/integration.md#supported-models)).

Version 1.0.0. Supported: Linux x86-64 and Windows x64 (MSVC), each with a CUDA 12 and a CUDA 13 build, as a full package and a `-system-cuda` package. The full support matrix is in [docs/integration.md](docs/integration.md#support-status).

- [docs/integration.md](docs/integration.md): building the engine into an application. Start here.
- [docs/c-api.md](docs/c-api.md): the C library, `include/mossinfer/mossinfer.h`.
- [docs/server-protocol.md](docs/server-protocol.md): `moss-server`, the JSON-lines sidecar process.

Build (CUDA 12.x or 13.x, CMake ≥ 3.28, Ninja; Linux: GCC ≥ 12 and GoogleTest; Windows: a Visual Studio 2022 x64 developer prompt):

```sh
cmake --preset linux-cuda13 && cmake --build --preset linux-cuda13
CUDA_VISIBLE_DEVICES="" ctest --preset linux-cuda13 -L '^cpu$'
```

The other presets are `linux-cuda12`, `windows-msvc-cuda13` and `windows-msvc-cuda12` ([docs/integration.md](docs/integration.md#getting-the-binaries)).

Licensed under Apache-2.0 (`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).
