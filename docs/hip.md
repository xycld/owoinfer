# HIP backend

`moss-infer-hip` runs MOSS-Image on AMD Radeon GPUs through ROCm. It is a standalone program in its own package
(`mossinfer-VERSION-linux-hip-x86_64.tar.gz`) for Linux x86-64; it needs ROCm 7 and neither CUDA nor an NVIDIA driver.
On a supported NVIDIA GPU, use `moss-infer`; on a GPU this program does not support, `moss-infer-vulkan`
([vulkan.md](vulkan.md)).

It runs txt2img with the official Base and SFT bf16 files, the official euler sampler and schedule and classifier-free
guidance with the uncond skip layers. The tier is `reference`. Everything else is refused with a message that names
the option: other samplers and schedulers, LoRAs, img2img and inpainting, hires fix, control adapters, references,
the lossy accelerators, quantized or GGUF files, and the CUDA engine's other memory options. The tokenizer, the seed's
initial noise and the schedule are the engine's own, so a seed gives the same inputs as on CUDA.

```sh
moss-infer-hip --model MOSS-Image/base --prompt "a red apple on a wooden table" --size 1024x1024 -o apple.png
moss-infer-hip --list-devices
moss-infer-hip --version
```

## Supported GPUs

| Architecture | GPUs | Validated |
|---|---|---|
| `gfx1100` (RDNA3) | Radeon RX 7900 XTX, 7900 XT, 7900 GRE; Radeon PRO W7900, W7800 | no: built only |
| `gfx1200` (RDNA4) | Radeon RX 9060 XT, 9060 | yes: RX 9060 XT |
| `gfx1201` (RDNA4) | Radeon RX 9070 XT, 9070, 9070 GRE; Radeon AI PRO R9700 | no: built only |

Built for gfx1100, gfx1200 and gfx1201; validated on gfx1200 (RX 9060 XT). The kernels run on the 16x16x16 bf16 matrix
instructions of these architectures. The package's binary holds code for all three; other AMD GPUs, the other RDNA3
chips (`gfx1101`, `gfx1102`: RX 7800 XT, 7700 XT, 7600) included, are listed with the reason and never used. The
DiT's weights are 9.5 GiB in bf16; a GPU with less memory streams part of them (see [Memory](#memory)).

## Install

ROCm 7 must be installed (the HIP runtime, `libamdhip64.so.7`), as AMD's ROCm installation guide describes for the
distribution, with the user in the `render` and `video` groups. The program finds the runtime on `LD_LIBRARY_PATH`,
then in `/opt/rocm/lib`, then in the system's library paths: for a ROCm installed elsewhere, put its `lib` directory on
`LD_LIBRARY_PATH`. Without the runtime the program does not start (the loader reports `libamdhip64.so.7` missing).
Unpack the archive anywhere; it holds `bin/moss-infer-hip` and the licence files.

## Usage

The flags are those of `moss-infer generate` that the backend runs, with the same defaults and checks:

| Flag | |
|---|---|
| `--model DIR` | the MOSS-Image Base or SFT folder (required) |
| `--prompt TEXT`, `--negative TEXT` | the prompts (`--prompt` required) |
| `-o FILE.png`, `--latent-out FILE.npy` | the image, the final latents (at least one; without `-o` the VAE decode is skipped) |
| `--size WxH`, `--width N`, `--height N` | multiples of 16, at most 4096 a side and 4.2 megapixels (default 1024x1024) |
| `--steps N`, `--cfg X`, `--shift X`, `--seed N` | default 50, 3, 5, 42; `--seed -1` picks a random seed (below) |
| `--skip-layers "22"`, `--max-seq N` | the uncond row's skipped blocks, the text tokens per prompt (default 4096) |
| `--tier reference` | accepted; the other tiers are refused |
| `--vram-budget GiB` | caps the device memory of the VAE and the DiT ([Memory](#memory)) |
| `--device N` | a HIP device (default: the first one the program supports) |
| `--no-metadata`, `--quiet` | no generation record in the PNG; no progress |
| `--list-devices`, `--version`, `--help` | the devices and whether each is supported; the version and the architectures built in; the usage |

The device and its kernels are printed at the start of every run. Progress goes to stderr one line per event:
`text encoder`, `denoise <step>/<steps> <seconds>`, `vae`, then a `done:` summary with each stage's time and how many of
the DiT's 42 blocks stayed on the device. The exit status is 0 on success, 2 for a usage error or an unsupported
request, 1 for any other failure. The seed that ran is in the `done:` line (not with `--quiet`) and in the PNG's
metadata (not with `--no-metadata`). The PNG's `generation` record has the fields of `moss-infer`'s without `attention`
and `dit_linears`, plus `noise_layout` and `backend: hip <architecture>`.

## Picking the program

A frontend picks the program per machine; the engine's packages do not include this one. Probe the CUDA engine first
(`moss-infer`'s `hello` or `mi_probe()`, [integration.md](integration.md#picking-the-build-cu12-or-cu13)): with a
recommended flavour, use it. Otherwise run `moss-infer-hip --list-devices`: a device followed by a `kernels:` line runs
it, and `--device` picks it when there are several. A device listed as `unusable`, no device (`no HIP devices`) or a
program that does not start (no ROCm 7 runtime) means `moss-infer-vulkan`, which needs only a Vulkan driver.

## Memory

The text encoder streams its 32 layers from the model files, at most two on the device at a time, and is freed before
the DiT loads. The VAE's weights (0.1 GiB) load first and stay; its decode needs 1.5 GiB at 1024x1024, 3.0 GiB at
1456x1456 and 6.0 GiB at 2048x2048 more, after the DiT is freed.

The DiT keeps as many of its 42 blocks on the device as fit beside the run's activations and a 0.5 GiB reserve: by
default in the device's free memory, with `--vram-budget` within that budget less the VAE's weights. The other blocks
stay in pinned host memory (about 0.21 GiB each, 8.8 GiB for all 42) and are copied into one of two device slots while
the block before them runs, every step. The DiT's embedders, modulations and text refiner and the two slots stay on the
device whatever the budget, about 1.03 GiB. A streamed run gives the resident run's result bit for bit (on the RX 9060
XT: whole runs at 512x512 with 5 blocks resident and at 1024x1024 with 2 and 17). With `--vram-budget` the VAE's decode must also fit
in the budget.

A run the device or the budget cannot hold is refused before the text encoder runs, with the sizes: a decode that does
not fit, activations that do not fit, a budget below the DiT's minimum, or less available host memory than the
streamed blocks pin. The DiT's part is checked again with the memory free when it loads.

## Speed and quality

On a Radeon RX 9060 XT (16 GiB), Base, 1024x1024, 50 steps, cfg 3:

| Stage | Time |
|---|---|
| Text encoder | 4.1 s |
| DiT load | 5.6 s |
| Denoise | 4.24 s per step, 212 s |
| VAE decode | 0.9 s |

With `--vram-budget 6` (17 of 42 blocks on the device) a step takes 4.43 s, with `--vram-budget 3` (2 blocks) 4.57 s.
A 512x512 step takes 0.71 s, a 1456x1456 step 14.3 s.

Against the official pipeline's golden case (fridge prompt, seed 42): the DiT's first step gives cosine 0.99998 on
both the cond and the uncond row, the text encoder stays within 2.5 times torch's own kernel-swap noise on every
golden prompt, and the DiT alone, run 50 steps on the golden text-encoder output, ends at cosine 0.987 of the golden
final latent. End to end the final latent is at 0.957, the same image with small layout shifts: at cfg 3 the sampler
amplifies any rounding difference in the text encoder's output (the CUDA engine's DiT fed torch's text-encoder output,
at cosine 0.99992 to the engine's own, ends at cosine 0.97 of the engine's own run).

## Building

`hip/` is its own CMake project (CMake 3.28 or newer) and needs ROCm 7 with its clang and the rocWMMA headers, and no
CUDA. `HIP_TARGET` lists the architectures built into the binary; the release package has all three.

```sh
cmake -S hip -B build-hip -G Ninja "-DHIP_TARGET=gfx1100;gfx1200;gfx1201" -DROCM_PATH=/opt/rocm
cmake --build build-hip
cpack --config build-hip/CPackConfig.cmake -B build-hip
```

Warnings are errors. The build compiles each kernel source once more per architecture, device code only, and checks
its assembly and code object (`hip/check_device.sh`): each kernel's count of bf16 matrix instructions, wave and
workgroup size 32, no scratch, spills or dynamic stack, and no undefined symbols. `-DMOSSINFER_STATIC_LIBSTDCXX=ON`
links libstdc++ statically, as the release package does.

With the tools on (the default, `MOSSINFER_HIP_TOOLS`) it also builds the backend's checks: the runtime's
(`hip_check`), the kernels' against host replays of their operation order (`hip_dit_kernel_check`,
`hip_te_kernel_check`, `hip_vae_check`), and the text encoder, DiT and VAE against the official pipeline's golden data
(`hip_te_check`, `hip_dit_run`, `hip_vae_check`). See [hip/README.md](../hip/README.md). `-DHIP_TARGET=nvidia` builds
the same sources for a CUDA device through HIP's NVIDIA headers, with a scalar matrix leaf: a logic check, not a
product.
