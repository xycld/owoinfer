# Vulkan backend

`moss-infer generate --backend vulkan` runs MOSS-Image on Vulkan instead of CUDA. It is the fallback for GPUs the CUDA
engine does not support (AMD, Intel, older NVIDIA cards) and is much slower than CUDA. On a supported NVIDIA GPU, use
the default CUDA backend.

It runs txt2img with the official Base and SFT bf16 files, the official euler sampler and schedule and classifier-free
guidance with the uncond skip layers. The tier is `reference`. Everything else is refused with a message that names
the option: other samplers and schedulers, LoRAs, img2img and inpainting, hires fix, control adapters, references,
the lossy accelerators, quantized or GGUF files, and the CUDA memory options. The tokenizer, the seed's initial noise
and the schedule are the engine's own, so a seed gives the same inputs as on CUDA.

```sh
moss-infer generate --backend vulkan --tier reference --model MOSS-Image/base \
  --prompt "a red apple on a wooden table" --size 1024x1024 -o apple.png
```

`--vulkan-device N` picks a device (default: the first usable one, a discrete GPU first). `--vulkan-kernels coop|plain`
picks the kernels (default: `coop` where the device has them). The device and kernels are printed at the start of
every run.

## Kernels

| Kernels | Needs | Used for |
|---|---|---|
| `coop` | Vulkan 1.3, `VK_KHR_cooperative_matrix` and `VK_KHR_shader_bfloat16` with 16x16x16 bf16 fragments, a compute subgroup size fixable at 32, 40 KiB of shared memory | GEMMs, attention and the VAE's convolutions on matrix cores |
| `plain` | Vulkan 1.2, buffer device addresses, 64-bit integers, 32 KiB of shared memory | every other device; no cooperative matrices, 16-bit arithmetic or particular subgroup size |

## Memory

The text encoder streams its 32 layers from the model files, one layer on the device at a time. The DiT keeps as many
of its 42 blocks on the device as fit next to the run's activations, from the driver's memory budget
(`VK_EXT_memory_budget`). The other blocks stay in host memory and are copied on the transfer queue while the block
before them runs. So a GPU with less than the DiT's 9.5 GiB still runs it, at the cost of host memory and bus traffic.
The VAE decodes the whole image at once: about 1.6 GiB at 1024x1024, 3.1 GiB at 1456x1456 and 6 GiB at 2048x2048. A
decode the device cannot hold is refused before the text encoder runs. The text encoder and the DiT are on the device
one after the other, never together.

Long operations are split into submits of bounded work, so no submit runs long enough to trip the Windows display
driver's watchdog on a slow GPU.

## Accuracy

Against the official PyTorch pipeline's golden data at 1024x1024 (`tests/golden/g_fridge_1024`), on the `coop` path:

| Stage | Result |
|---|---|
| Text encoder, last hidden state | cosine 0.99988 (prompt), 0.99980 (negative prompt); torch itself with one kernel swapped: 0.99989, 0.99983 |
| DiT, step 0 prediction | cosine 0.999976 (cond), 0.999983 (uncond); the CUDA engine's test asks for 0.99988 |
| VAE decode of the golden latent | PSNR 69.4 dB against the golden PNG, 99.25 % identical bytes, max difference 1 |

Whole images keep the composition and the text but not the pixels: 50 sampling steps magnify any difference. The
golden 1024x1024 case through `moss-infer` ends with a final latent at cosine 0.974 to the official pipeline's (the CUDA
backend: 0.992, being nearly bit-identical to torch op by op) and an image at 30.8 dB PSNR. That is the sampler's
sensitivity, not an error of the backend: the same Vulkan steps run on torch's own text embeddings instead of the Vulkan
encoder's (cosine 0.9998 apart, about torch's own variation when one of its kernels changes) end 0.971 apart from each
other. A 512x512, 20-step image with seed 7 is 45.97 dB PSNR from the CUDA backend's image of the same command.

## Building

The backend is off by default. It needs Vulkan headers that define `VK_KHR_shader_bfloat16` and a `glslc` that compiles
`GL_EXT_bfloat16`, as in a current Vulkan SDK. The binaries load the Vulkan loader at run time, so a build with the
backend still starts on a machine without Vulkan and only `--backend vulkan` reports it missing.

```sh
cmake --preset linux-cuda13 -DMOSSINFER_VULKAN=ON
cmake --build --preset linux-cuda13
```

The backend's own checks and benchmarks build from `vulkan/` alone, without CUDA; see [vulkan/README.md](../vulkan/README.md).
