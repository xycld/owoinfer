// Bit-exact reimplementation of torch's CPU-generator randn, so user seeds reproduce the official
// app's initial noise (randn_tensor(..., generator=torch.Generator("cpu").manual_seed(seed))).
//
// Reference: torch 2.14.1 x86-64 wheel (GCC 13 build), CPU capability AVX2. What it does:
//  - Generator: mt19937 seeded with the low 32 bits of the seed (seed 2^32 == seed 0).
//  - numel < 16: per element at::normal_distribution<double> on random64() uniforms, Box-Muller with
//    glibc log1p/sincos, the sin half cached for the next element, then rounded to float.
//  - numel >= 16: all numel float uniforms ((u32 & 0xffffff) * 2^-24) are drawn first, Box-Muller runs
//    in place on 16-element chunks (lanes j and j+8 pair up), and when numel % 16 != 0 the last 16
//    elements are redrawn and transformed again. The chunk transform is avx_mathfun.h's Cephes
//    log256_ps/sincos256_ps, not Sleef. GCC contracted many of its mul+add pairs into FMAs, and that
//    changes the results, so the port follows the FMA placement in libtorch_cpu.so's disassembly.
//
// Matches torch bit for bit on x86-64 CPUs with AVX2 or AVX512 (AVX512 CPUs fall back to the AVX2
// kernel because torch builds no AVX512 one). Not reproduced: torch on a CPU without AVX2 (scalar
// logf/cos/sin kernel), other architectures, and other torch builds whose compiler contracts the
// kernel differently. The numel < 16 path calls the platform libm, so it matches only where that
// libm agrees with glibc; the engine always draws >= 128 values and never takes that path.
#pragma once

#include <cstdint>
#include <vector>

namespace mi::rng {

// Same values as torch.randn(numel, generator=torch.Generator("cpu").manual_seed(seed),
// dtype=torch.float32). Throws std::invalid_argument if numel < 0.
std::vector<float> torch_cpu_randn(std::uint64_t seed, std::int64_t numel);

}  // namespace mi::rng
