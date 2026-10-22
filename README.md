# MossInfer

A native C++/CUDA inference engine for the MOSS-Image text-to-image model (5B DiT, Qwen3.5 text encoder, FLUX.2 VAE), with no PyTorch or Python at run time. It plans memory against the GPU it finds and streams the model when it does not fit.

Pre-release (0.2.0). Supported now: Linux x86-64 with CUDA 13. Planned: Windows and CUDA 12 builds. The full support matrix is in [docs/integration.md](docs/integration.md#support-status).

- [docs/integration.md](docs/integration.md): building the engine into an application. Start here.
- [docs/c-api.md](docs/c-api.md): the C library, `include/mossinfer/mossinfer.h`.
- [docs/server-protocol.md](docs/server-protocol.md): `moss-server`, the JSON-lines sidecar process.

Build (Linux, CUDA 13.x, CMake ≥ 3.28, Ninja, GCC ≥ 12, GoogleTest):

```sh
cmake --preset linux-cuda13 && cmake --build --preset linux-cuda13
CUDA_VISIBLE_DEVICES="" ctest --preset linux-cuda13 -L '^cpu$'
```

Licensed under Apache-2.0 (`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`).
