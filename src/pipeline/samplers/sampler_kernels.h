// Internal to the sampler driver: GPU coefficient evaluation and the per-step update kernels. Pointers are device
// fp32 arrays of n elements; every op rounds once (explicitly rounded intrinsics, no contraction).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace mi::pipeline::sampler_kernels {

// Layout of the per-step coefficients (floats per step) and their meaning, per algorithm. Flags are 0 or 1.
enum EulerCoef { kEulS, kEulDt, kEulCount };
enum AncestralCoef { kAncFinal, kAncRatio, kAncKeep, kAncAlpha, kAncRenoise, kAncCount };
enum HeunCoef { kHeuS, kHeuSNext, kHeuDt, kHeuCount };
enum Dpm2mCoef { kDpmC1, kDpmC2, kDpmA, kDpmQ, kDpmCount };
enum SdeCoef { kSdeFinal, kSdeC1, kSdeC2, kSdeC3, kSdeNoise, kSdeSNext, kSdeCount };
enum ErCoef { kErFinal, kErC1, kErC2, kErS2, kErDiv2, kErS3, kErDiv3, kErNoise, kErAlpha, kErCount };
enum ResCoef { kResS, kResDt, kResExp, kResH, kResB1, kResB2, kResCount };

enum class CoefKind { kEuler, kAncestral, kHeun, kDpm2m, kSde, kEr, kRes };

// Evaluates the coefficients of all steps on the GPU: `sigmas` (device, steps + 1 values) -> `coef` (device,
// steps * count). eta as in SamplerOptions, eta_plus_one = fp32(eta + 1) (a Python-float sum in the reference).
void compute_coefficients(CoefKind kind, const float* sigmas, int steps, float eta, float eta_plus_one, float* coef,
                          cudaStream_t stream);

// y = expm1(x) exactly as torch's CUDA expm1 (exposed for tests).
void torch_expm1(const float* x, float* y, std::int64_t n, cudaStream_t stream);

void euler(float* x, const float* den, float s, float dt, std::int64_t n, cudaStream_t stream);
void velocity_euler(float* x, const float* v, float h, std::int64_t n, cudaStream_t stream);
void ab2(float* x, const float* v, const float* v_prev, float h, float a, float b, std::int64_t n, cudaStream_t stream);
// noise may be null (no noise term).
void ancestral(float* x, const float* den, const float* noise, float ratio, float keep, float alpha, float renoise,
               float s_noise, std::int64_t n, cudaStream_t stream);
// d = (x - den) / s, x2 = x + d * dt
void heun_predict(const float* x, const float* den, float s, float dt, float* d, float* x2, std::int64_t n,
                  cudaStream_t stream);
// x += ((d + (x2 - den2) / s_next) / 2) * dt
void heun_correct(float* x, const float* d, const float* x2, const float* den2, float s_next, float dt, std::int64_t n,
                  cudaStream_t stream);
// old null: x = c1 x - c2 den; else x = c1 x - c2 (a den - q old)
void dpm2m(float* x, const float* den, const float* old, float c1, float c2, float a, float q, std::int64_t n,
           cudaStream_t stream);
// x = c1 x + c2 den; old: x += c3 (den - old); noise: x += ((noise * s_next) * c_noise) * s_noise
void dpm2m_sde(float* x, const float* den, const float* old, const float* noise, float c1, float c2, float c3,
               float s_next, float c_noise, float s_noise, std::int64_t n, cudaStream_t stream);
// x = c1 x + c2 den; stage >= 2: dd = (den - old) / div2, x += s2 dd; stage 3: du = (dd - old_dd) / div3,
// x += s3 du; noise: x += ((alpha * noise) * s_noise) * c_noise. dd is written for the next step.
void er_sde(float* x, const float* den, const float* old, float* dd, const float* old_dd, const float* noise, int stage,
            float c1, float c2, float s2, float div2, float s3, float div3, float alpha, float s_noise, float c_noise,
            std::int64_t n, cudaStream_t stream);
// x = e x + h (b1 den + b2 old)
void res_multistep(float* x, const float* den, const float* old, float e, float h, float b1, float b2, std::int64_t n,
                   cudaStream_t stream);
// x_in[o, p] = z[o, p] + sum_c q[o, c] x0[c, p] over [channels, pixels]
void dataspace_mix(const float* z, const float* x0, const float* q, int channels, std::int64_t pixels, float* x_in,
                   cudaStream_t stream);
// x0 = z - v
void dataspace_update(float* x0, const float* z, const float* v, std::int64_t n, cudaStream_t stream);

// Official guidance on velocities: out = u + g * (c - u); out = p + s * (c - k).
void cfg_velocity(const float* c, const float* u, float g, float* out, std::int64_t n, cudaStream_t stream);
void slg_velocity(const float* p, const float* c, const float* k, float s, float* out, std::int64_t n, cudaStream_t stream);

void noise_scaling(float* noise, const float* latent, float sigma, std::int64_t n, cudaStream_t stream);
void velocity_to_denoised(const float* x, const float* v, float sigma, float* out, std::int64_t n, cudaStream_t stream);
void cfg_denoised(const float* x, const float* vc, const float* vu, float sigma, float g, float* out, std::int64_t n,
                  cudaStream_t stream);
void slg_denoised(const float* x, const float* vc, const float* vs, float sigma, float scale, float* out, std::int64_t n,
                  cudaStream_t stream);

}  // namespace mi::pipeline::sampler_kernels
