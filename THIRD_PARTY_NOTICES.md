# Third-party notices

MossInfer contains, or is derived in part from, the third-party components below. Each vendored directory keeps
the upstream licence file; its `VENDORED.md` lists what was kept and what was changed.

| Component | Licence | Copyright | Upstream (commit) | In this repository |
|---|---|---|---|---|
| CUTLASS / CuTe (C++ headers) | BSD-3-Clause | Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES | https://github.com/NVIDIA/cutlass, v4.7.1 (`cb4247394dd82148787aed73e5dc7cef33cbf862`) | `third_party/cutlass/` (licence: `LICENSE.txt`). `UpGateScaleBroadcast` in `src/kernels/quant/int8_gemm_impl.cuh` derives from its `VisitorRowBroadcast`; `LoraGemmKernel` and `VisitorKernel` in `src/kernels/lora/lora_gemm_kernel.cuh` are adapted from its `GemmWithEpilogueVisitor` and `DefaultGemmWithVisitor` (Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES); `src/kernels/nvfp4/gemm_sm120a_impl.cuh` follows its example `79a_blackwell_geforce_nvfp4_bf16_gemm.cu` |
| FlashAttention-2 (forward kernel headers) | BSD-3-Clause | Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file (Tri Dao) | https://github.com/Dao-AILab/flash-attention, v2.8.3 (`060c9188beec3a8b62b33a3bfa6d5d2d44975fab`) | `third_party/flash-attention/` (`LICENSE`, `AUTHORS`), launched from `src/kernels/attention/` |
| SageAttention (CUDA headers) | Apache-2.0 | Copyright 2024 Jintao Zhang, Haofeng Huang (file headers: Copyright (c) 2024 by SageAttention team). Embedded, also Apache-2.0: FlashInfer team 2023, adapted from flashinfer v0.1.5 (`cp_async.cuh`, `math.cuh`, `mma.cuh`, `permuted_smem.cuh`); vLLM team 2023 and NVIDIA 2020–2023 from FasterTransformer (`reduction_utils.cuh`) | https://github.com/thu-ml/SageAttention (`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`) | `third_party/sageattention/` (`LICENSE`, identical to upstream's). `attn_utils.cuh`, `qk_int_sv_f8_cuda_sm89.cuh` and `qk_int_sv_f16_cuda_sm80.cuh` are modified and each carries a "MossInfer modifications" notice. Launchers in `src/kernels/attention_q8/`; `q8_preprocess.cu` there derives from upstream `csrc/fused/fused.cu`. Upstream has no NOTICE file |
| comfy-kitchen | Apache-2.0 | Copyright (c) 2025 Comfy Org | https://github.com/Comfy-Org/comfy-kitchen (`dba9fac464e93143146f9bafa4f9cb0a9a116bc5`) | `src/kernels/quant/int8_gemm_impl.cuh` is adapted from `comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu` (modified; the file says how). `src/kernels/quant/` and `src/weights/quant/` reproduce its INT8 ConvRot math (rotation order, quantization, dequantization). `src/kernels/quant4/int4_gemm_impl.cuh` adapts `FusedInt4Gemm` from `comfy_kitchen/backends/cuda/ops/convrot_w4a4.cu`, and `src/kernels/quant4/` with `src/weights/quant/convrot_w4a4.*` reproduce its W4A4 ConvRot quantization math |
| MOSS-Image reference pipeline; diffusers `FlowMatchEulerDiscreteScheduler` | Apache-2.0 | Copyright OpenMOSS-Team (MOSS-Image); Copyright The HuggingFace Team (diffusers) | https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base, https://github.com/huggingface/diffusers | `src/pipeline/sampler.*` and `src/pipeline/samplers/schedules.cpp` (the `moss` schedule, the velocity Euler step and classifier-free guidance on velocities) reimplement their sampling schedule and operation order. No code is copied |
| k-diffusion | MIT | Copyright (c) 2022 Katherine Crowson | https://github.com/crowsonkb/k-diffusion | `src/pipeline/samplers/` reimplements its samplers (euler, heun, dpmpp_2m, dpmpp_2m_sde) and the karras and exponential schedules, matching their operation order. No code is copied |
| ER-SDE-Solver | MIT | Copyright (c) 2023 Qinpeng Cui | https://github.com/QinpengCui/ER-SDE-Solver | `src/pipeline/samplers/sampler_kernels.cu` (`er_sde`) implements its method (arXiv 2309.06169; noise scaler, quadrature); no code is copied |
| torchsde (`BrownianInterval`) | Apache-2.0 | Copyright 2020 Google LLC | https://github.com/google-research/torchsde | `src/pipeline/samplers/brownian_tree.*` is adapted from its Brownian interval (modified; the file header says how) |
| NumPy (`SeedSequence`) | BSD-3-Clause | Copyright (c) 2005-2026, NumPy Developers | https://github.com/numpy/numpy | `src/pipeline/samplers/brownian_tree.cpp` reproduces its SeedSequence entropy-mixing constants and algorithm |
| PyTorch (Philox `randn`, `linspace`, reduction order) | BSD-3-Clause | Copyright (c) 2016- Facebook, Inc. and contributors | https://github.com/pytorch/pytorch | `src/pipeline/samplers/torch_cuda_randn.*`, `src/pipeline/rng_torch_compat.*` and helpers in `sampler_kernels.cu` reproduce its random-number and arithmetic order |
| LyCORIS | Apache-2.0 | Copyright (c) KohakuBlueleaf and contributors | https://github.com/KohakuBlueleaf/LyCORIS | `src/models/lora/` follows its LoHa/LoKr weight definitions and scaling |
| stb_image | MIT or public domain (Unlicense), at your choice | Copyright (c) 2017 Sean Barrett | https://github.com/nothings/stb (`013ac3beddff3dbffafd5177e7972067cd2b5083`) | `third_party/stb/` (`LICENSE`). Linked into the engine for PNG/JPEG decoding of init images and masks (`src/pipeline/pipeline_image.cpp`), and used by the tests |
| ggml / llama.cpp (GGUF container, ggml block types) | MIT | Copyright (c) 2023-2026 The ggml authors | https://github.com/ggml-org/ggml, https://github.com/ggml-org/llama.cpp (`gguf-py`) | `src/weights/gguf.*` reads the GGUF format; its dequantization of the Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K blocks is derived from ggml's block layouts and dequantization (`dequantize_row_*`, their operation order). `tests/data/gguf/` are outputs of the `gguf` Python package. Licence text below |
| xxHash (XXH64 algorithm and constants) | BSD-2-Clause | Copyright (c) Yann Collet | https://github.com/Cyan4973/xxHash (specification) | `src/util/xxh64.h` is an independent implementation from the specification |
| Unicode tables of HF `tokenizers` 0.23.3 | Apache-2.0 (tokenizers); Unicode License v3 (Unicode Character Database) | Copyright (c) Hugging Face, Inc.; Copyright (c) Unicode, Inc. | https://github.com/huggingface/tokenizers | `src/tokenizer/unicode_tables.inc` (generated data, measured from the library by `tools/gen/gen_unicode_tables.py`) |

`third_party/reference/` is a local, untracked checkout of reference code (the official MOSS-Image pipeline and
ComfyUI nodes) used to generate test vectors. It is not part of the repository or of any distribution.

## Reference implementations used as test oracles (not included)

The standard samplers and schedulers in `src/pipeline/samplers/` were checked bit for bit against the outputs of
ComfyUI (https://github.com/comfyanonymous/ComfyUI, GPL-3.0), used as a test oracle on a local, untracked checkout.
No ComfyUI code is included in MossInfer. The test vectors under `tests/data/samplers/` are program outputs. Where
our code reproduces a numeric operation order, that order is a functional requirement for bit-exact results, and the
underlying methods are published (Karras et al. 2022; Lu et al. 2022, DPM-Solver++; Cui et al. 2023, ER-SDE-Solver;
Zhang et al. 2023, arXiv 2308.02157). The same applies to the official MOSS-Image pipeline and diffusers, used as
references for exactness.

The NVFP4 codec and reader (`src/weights/quant/nvfp4_codec.*`, the `nvfp4` case of `src/weights/quant/quant_model.*`)
are written from the format definition (NVIDIA's NVFP4 description, the cuBLAS 1D block-scaling layout and CUTLASS's
`Sm1xxBlockScaledConfig`) and ComfyUI's public `comfy_quant` `nvfp4` file convention. They were checked bit for bit
against comfy-kitchen's eager NVFP4 functions, used only as a test oracle; `tests/data/nvfp4/` holds program outputs.

## comfy-kitchen NOTICE

The upstream NOTICE file reads, in the part that applies to the code used here:

```
ComfyKitchen

Copyright (c) 2025 Comfy Org. All rights reserved.

This product includes software developed at NVIDIA CORPORATION.
```

Its other entries credit dependencies of the upstream Python package (CUDA runtime, PyTorch, nanobind,
FlashAttention, PyTorch AO and SageAttention). None of their code comes into MossInfer through comfy-kitchen.
FlashAttention-2 and SageAttention are vendored directly; see the table above.

## Licence texts

- BSD-3-Clause: `third_party/cutlass/LICENSE.txt`, `third_party/flash-attention/LICENSE`.
- Apache-2.0: `third_party/sageattention/LICENSE`. comfy-kitchen, torchsde, LyCORIS and HF `tokenizers` use the same licence text,
  https://www.apache.org/licenses/LICENSE-2.0.
- MIT / Unlicense: `third_party/stb/LICENSE`.
- BSD-2-Clause (xxHash), MIT (ggml) and Unicode License v3: reproduced below, since they have no file in this repository.

## xxHash licence (BSD-2-Clause)

From https://github.com/Cyan4973/xxHash/blob/dev/LICENSE:

```
xxHash Library
Copyright (c) 2012-2021 Yann Collet
All rights reserved.

BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## ggml licence (MIT)

From https://github.com/ggml-org/ggml/blob/master/LICENSE:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode License v3

From https://www.unicode.org/license.txt:

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```
