// Sampler driver kernels. Each torch op of the reference rounds once, so every op here is an explicitly rounded
// intrinsic (nvcc would otherwise contract mul + add into FMAs). Scalar coefficients follow the reference's 0-dim
// CUDA tensor ops: CUDA math library logf/expf/powf (as torch's kernels call them, no fast-math), torch's expm1, IEEE
// division and sqrt; a tensor divided by a Python number is a multiplication by its fp32 reciprocal in torch, a
// tensor divided by a tensor a true division.
#include "pipeline/samplers/sampler_kernels.h"

#include <cmath>

#include "core/cuda_check.h"

namespace mi::pipeline::sampler_kernels {
namespace {

constexpr int kThreads = 256;

unsigned blocks(std::int64_t n) {
  const std::int64_t b = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(b < 65535 * 8 ? b : 65535 * 8);
}

#define MI_GRID_LOOP(i, n)                                                                  \
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)

__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }

// expm1 as torch's CUDA kernels compute it: pinned to the CUDA 13.0 math library's expm1f (transcribed from the SASS
// of torch 2.14.1+cu130's expm1 kernel). Later CUDA releases changed expm1f (CUDA 13.4 differs on ~1 % of inputs by
// 1 ulp), so a torch built with another CUDA may differ again; the tests check the vectors' torch build. The power of two is formed exactly (the hardware's ex2 is exact for integers; the
// paths where it is not are overridden by the final selects).
__device__ float torch_expm1f(float x) {
  const float t = rintf(__fmul_rn(x, 1.4426950216293334961f));
  float j = !(fabsf(x) < 0.40999999642372131348f) ? t : 0.0f;
  const bool j_is_128 = j == 128.0f;
  float f = __fmaf_rn(-j, 0.693145751953125f, x);
  f = __fmaf_rn(-j, 1.428606765330187045e-06f, f);
  float u = __fmaf_rn(f, __int_as_float(0x3ab5ebe6), 0.0083824126049876213074f);
  if (j_is_128) j = __fadd_rn(j, -1.0f);
  u = __fmaf_rn(f, u, 0.041667830199003219604f);
  const bool normal = !(j < -126.0f);
  u = __fmaf_rn(f, u, 0.16666397452354431152f);
  u = __fmaf_rn(f, u, 0.49999994039535522461f);
  float s;
  if (normal && j <= 127.0f) {
    s = __int_as_float((static_cast<int>(j) + 127) << 23);
  } else if (normal) {
    s = INFINITY;  // j > 127: overflow, selected to +inf below
  } else {
    s = 0.0f;      // j < -126: the result is selected to -1 below
  }
  u = __fmul_rn(f, u);
  u = __fmaf_rn(f, u, f);
  float r = __fadd_rn(s, -1.0f);
  r = __fmaf_rn(u, s, r);
  if (j_is_128) r = __fadd_rn(r, r);
  if (j > 128.0f) r = INFINITY;
  if (j < -25.0f) r = -1.0f;
  if (x == 0.0f) r = __fadd_rn(x, x);
  return r;
}

// -log(sigma) (dpmpp_2m, res_multistep t_fn) and the flow half-log-SNR -logit(sigma) = -log(sigma / (1 - sigma)).
__device__ float t_fn(float s) { return -logf(s); }
__device__ float half_log_snr(float s) { return -logf(div(s, sub(1.0f, s))); }
// torch.nan_to_num(x, nan=0.0) for the finite values these coefficients take (infinities pass through).
__device__ float nan0(float x) { return isnan(x) ? 0.0f : x; }
// er_sde's default noise scaler x * (exp(x ** 0.3) + 10).
__device__ float er_scaler(float x) { return mul(x, add(expf(powf(x, 0.3f)), 10.0f)); }

// torch.sum of a contiguous, 16-byte aligned fp32 CUDA tensor of 200 elements, in torch's order (Reduce.cuh, input
// vectorized by 4 over a 32-lane warp: lane l accumulates its vectors l and l + 32 into four partial sums, combined
// ((p0 + p1) + p2) + p3, then a shuffle-down tree with offsets 16, 8, 4, 2, 1). Needs torch >= 2.14: earlier
// releases ran the intra-warp tree with increasing offsets (1, 2, ..., 16), which sums in a different order.
__device__ float torch_sum200(const float* v) {
  constexpr int kN = 200, kLanes = 32, kVec = 4;
  float lane[kLanes];
  for (int l = 0; l < kLanes; ++l) {
    float p[kVec] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int idx = l; idx * kVec + kVec - 1 < kN; idx += kLanes) {
      for (int i = 0; i < kVec; ++i) p[i] = add(p[i], v[idx * kVec + i]);
    }
    lane[l] = add(add(add(p[0], p[1]), p[2]), p[3]);
  }
  for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
    for (int l = 0; l + offset < kLanes; ++l) lane[l] = add(lane[l], lane[l + offset]);
  }
  return lane[0];
}

__global__ void coef_kernel(CoefKind kind, const float* sig, int steps, float eta, float eta1, float* c) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int i = 0; i < steps; ++i) {
    const float s = sig[i], sn = sig[i + 1];
    switch (kind) {
      case CoefKind::kEuler: {
        float* o = c + i * kEulCount;
        o[kEulS] = s;
        o[kEulDt] = sub(sn, s);
        break;
      }
      case CoefKind::kHeun: {
        float* o = c + i * kHeuCount;
        o[kHeuS] = s;
        o[kHeuSNext] = sn;
        o[kHeuDt] = sub(sn, s);
        break;
      }
      case CoefKind::kAncestral: {
        // sample_euler_ancestral_RF
        float* o = c + i * kAncCount;
        o[kAncFinal] = sn == 0.0f ? 1.0f : 0.0f;
        const float ratio = add(1.0f, mul(sub(div(sn, s), 1.0f), eta));  // downstep_ratio
        const float sd = mul(sn, ratio);
        const float a1 = sub(1.0f, sn), ad = sub(1.0f, sd);
        const float r2 = div(sd, s);
        o[kAncRatio] = r2;
        o[kAncKeep] = sub(1.0f, r2);
        o[kAncAlpha] = div(a1, ad);
        o[kAncRenoise] = sqrtf(sub(mul(sn, sn), div(mul(mul(sd, sd), mul(a1, a1)), mul(ad, ad))));
        break;
      }
      case CoefKind::kDpm2m: {
        float* o = c + i * kDpmCount;
        const float t = t_fn(s), tn = t_fn(sn);
        const float h = sub(tn, t);
        o[kDpmC1] = div(expf(-tn), expf(-t));  // sigma_fn(t_next) / sigma_fn(t)
        o[kDpmC2] = torch_expm1f(-h);
        o[kDpmA] = 0.0f;
        o[kDpmQ] = 0.0f;
        if (i > 0) {
          const float h_last = sub(t, t_fn(sig[i - 1]));
          const float r = div(h_last, h);
          const float q = div(1.0f, mul(2.0f, r));  // 1 / (2 r) = reciprocal(2 r) * 1
          o[kDpmA] = add(1.0f, q);
          o[kDpmQ] = q;
        }
        break;
      }
      case CoefKind::kSde: {
        // sample_dpmpp_2m_sde, midpoint
        float* o = c + i * kSdeCount;
        o[kSdeFinal] = sn == 0.0f ? 1.0f : 0.0f;
        o[kSdeSNext] = sn;
        const float ls = half_log_snr(s), lt = half_log_snr(sn);
        const float h = sub(lt, ls);
        const float h_eta = mul(h, eta1);
        const float alpha_t = mul(sn, expf(lt));
        const float em = -torch_expm1f(-h_eta);
        o[kSdeC1] = mul(div(sn, s), expf(mul(-h, eta)));
        o[kSdeC2] = mul(alpha_t, em);
        o[kSdeC3] = 0.0f;
        if (i > 0) {
          const float h_last = sub(half_log_snr(s), half_log_snr(sig[i - 1]));
          const float r = div(h_last, h);
          o[kSdeC3] = mul(mul(mul(0.5f, alpha_t), em), div(1.0f, r));
        }
        o[kSdeNoise] = sqrtf(-torch_expm1f(mul(mul(-2.0f, h), eta)));
        break;
      }
      case CoefKind::kEr: {
        // sample_er_sde: er_lambda = exp(-half_log_snr) = exp(logit(sigma))
        float* o = c + i * kErCount;
        for (int k = 0; k < kErCount; ++k) o[k] = 0.0f;
        o[kErFinal] = sn == 0.0f ? 1.0f : 0.0f;
        const auto lam = [&](int k) { return expf(-half_log_snr(sig[k])); };
        const float ls = lam(i), lt = lam(i + 1);
        const float alpha_s = div(s, ls), alpha_t = div(sn, lt);
        const float r_alpha = div(alpha_t, alpha_s);
        const float r = div(er_scaler(lt), er_scaler(ls));
        o[kErAlpha] = alpha_t;
        o[kErC1] = mul(r_alpha, r);
        o[kErC2] = mul(alpha_t, sub(1.0f, r));
        if (i >= 1) {
          const float dt = sub(lt, ls);
          const float step = mul(-dt, 1.0f / 200.0f);  // -dt / num_integration_points
          float terms1[200], terms2[200];
          for (int j = 0; j < 200; ++j) {
            const float pos = add(lt, mul(static_cast<float>(j), step));  // lambda_pos
            const float scaled = er_scaler(pos);
            terms1[j] = div(1.0f, scaled);
            terms2[j] = div(sub(pos, ls), scaled);
          }
          const float s_1 = mul(torch_sum200(terms1), step);
          o[kErDiv2] = sub(ls, lam(i - 1));
          o[kErS2] = mul(alpha_t, add(dt, mul(s_1, er_scaler(lt))));
          if (i >= 2) {
            const float s_u = mul(torch_sum200(terms2), step);
            o[kErDiv3] = mul(sub(ls, lam(i - 2)), 0.5f);
            o[kErS3] = mul(alpha_t, add(mul(mul(dt, dt), 0.5f), mul(s_u, er_scaler(lt))));
          }
        }
        o[kErNoise] = nan0(sqrtf(sub(mul(lt, lt), mul(mul(ls, ls), mul(r, r)))));
        break;
      }
      case CoefKind::kRes: {
        // res_multistep with eta 0: sigma_down = sigma_next, old_sigma_down = sigma
        float* o = c + i * kResCount;
        for (int k = 0; k < kResCount; ++k) o[k] = 0.0f;
        o[kResS] = s;
        o[kResDt] = sub(sn, s);
        if (i > 0 && sn != 0.0f) {
          const float t = t_fn(s), t_old = t_fn(s), t_next = t_fn(sn), t_prev = t_fn(sig[i - 1]);
          const float h = sub(t_next, t);
          const float c2 = div(sub(t_prev, t_old), h);
          const float phi1 = div(torch_expm1f(-h), -h);
          const float phi2 = div(sub(phi1, 1.0f), -h);
          o[kResB1] = nan0(sub(phi1, div(phi2, c2)));
          o[kResB2] = nan0(div(phi2, c2));
          o[kResExp] = expf(-h);  // sigma_fn(h)
          o[kResH] = h;
        }
        break;
      }
    }
  }
}

__global__ void expm1_kernel(const float* x, float* y, std::int64_t n) {
  MI_GRID_LOOP(i, n) { y[i] = torch_expm1f(x[i]); }
}

__global__ void euler_kernel(float* x, const float* den, float s, float dt, std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float d = div(sub(x[i], den[i]), s);
    x[i] = add(x[i], mul(d, dt));
  }
}

__global__ void velocity_euler_kernel(float* x, const float* v, float h, std::int64_t n) {
  MI_GRID_LOOP(i, n) { x[i] = add(x[i], mul(h, v[i])); }
}

__global__ void ab2_kernel(float* x, const float* v, const float* v_prev, float h, float a, float b, std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float d = sub(mul(a, v[i]), mul(b, v_prev[i]));
    x[i] = add(x[i], mul(h, d));
  }
}

__global__ void ancestral_kernel(float* x, const float* den, const float* noise, float ratio, float keep, float alpha,
                                 float renoise, float s_noise, std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    float y = add(mul(ratio, x[i]), mul(keep, den[i]));
    if (noise != nullptr) y = add(mul(alpha, y), mul(mul(noise[i], s_noise), renoise));
    x[i] = y;
  }
}

__global__ void heun_predict_kernel(const float* x, const float* den, float s, float dt, float* d, float* x2,
                                    std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float di = div(sub(x[i], den[i]), s);
    d[i] = di;
    x2[i] = add(x[i], mul(di, dt));
  }
}

__global__ void heun_correct_kernel(float* x, const float* d, const float* x2, const float* den2, float s_next, float dt,
                                    std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float d2 = div(sub(x2[i], den2[i]), s_next);
    const float dp = mul(add(d[i], d2), 0.5f);  // (d + d2) / 2
    x[i] = add(x[i], mul(dp, dt));
  }
}

__global__ void dpm2m_kernel(float* x, const float* den, const float* old, float c1, float c2, float a, float q,
                             std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float dd = old == nullptr ? den[i] : sub(mul(a, den[i]), mul(q, old[i]));
    x[i] = sub(mul(c1, x[i]), mul(c2, dd));
  }
}

__global__ void dpm2m_sde_kernel(float* x, const float* den, const float* old, const float* noise, float c1, float c2,
                                 float c3, float s_next, float c_noise, float s_noise, std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    float y = add(mul(c1, x[i]), mul(c2, den[i]));
    if (old != nullptr) y = add(y, mul(c3, sub(den[i], old[i])));
    if (noise != nullptr) y = add(y, mul(mul(mul(noise[i], s_next), c_noise), s_noise));
    x[i] = y;
  }
}

__global__ void er_sde_kernel(float* x, const float* den, const float* old, float* dd, const float* old_dd,
                              const float* noise, int stage, float c1, float c2, float s2, float div2, float s3,
                              float div3, float alpha, float s_noise, float c_noise, std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    float y = add(mul(c1, x[i]), mul(c2, den[i]));
    if (stage >= 2) {
      const float ddi = div(sub(den[i], old[i]), div2);
      y = add(y, mul(s2, ddi));
      if (stage >= 3) {
        const float du = div(sub(ddi, old_dd[i]), div3);
        y = add(y, mul(s3, du));
      }
      dd[i] = ddi;
    }
    if (noise != nullptr) y = add(y, mul(mul(mul(alpha, noise[i]), s_noise), c_noise));
    x[i] = y;
  }
}

__global__ void res_kernel(float* x, const float* den, const float* old, float e, float h, float b1, float b2,
                           std::int64_t n) {
  MI_GRID_LOOP(i, n) { x[i] = add(mul(e, x[i]), mul(h, add(mul(b1, den[i]), mul(b2, old[i])))); }
}

__global__ void dataspace_mix_kernel(const float* z, const float* x0, const float* q, int channels, std::int64_t pixels,
                                     float* x_in) {
  const std::int64_t total = static_cast<std::int64_t>(channels) * pixels;
  MI_GRID_LOOP(i, total) {
    const std::int64_t o = i / pixels, p = i % pixels;
    float acc = 0.0f;
    for (int ch = 0; ch < channels; ++ch) acc = fmaf(q[o * channels + ch], x0[ch * pixels + p], acc);
    x_in[i] = add(z[i], acc);
  }
}

__global__ void sub_kernel(float* out, const float* a, const float* b, std::int64_t n) {
  MI_GRID_LOOP(i, n) { out[i] = sub(a[i], b[i]); }
}

__global__ void cfg_velocity_kernel(const float* c, const float* u, float g, float* out, std::int64_t n) {
  MI_GRID_LOOP(i, n) { out[i] = add(u[i], mul(g, sub(c[i], u[i]))); }
}

__global__ void slg_velocity_kernel(const float* p, const float* c, const float* k, float s, float* out, std::int64_t n) {
  MI_GRID_LOOP(i, n) { out[i] = add(p[i], mul(s, sub(c[i], k[i]))); }
}

__global__ void noise_scaling_kernel(float* noise, const float* latent, float sigma, std::int64_t n) {
  // sigma * (1.0 * noise) + (1 - sigma) * latent
  const float keep = sub(1.0f, sigma);
  MI_GRID_LOOP(i, n) { noise[i] = add(mul(sigma, noise[i]), mul(keep, latent == nullptr ? 0.0f : latent[i])); }
}

__global__ void v2d_kernel(const float* x, const float* v, float s, float* out, std::int64_t n) {
  MI_GRID_LOOP(i, n) { out[i] = sub(x[i], mul(v[i], s)); }
}

__global__ void cfg_den_kernel(const float* x, const float* vc, const float* vu, float s, float g, float* out,
                               std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float dc = sub(x[i], mul(vc[i], s)), du = sub(x[i], mul(vu[i], s));
    out[i] = add(du, mul(sub(dc, du), g));
  }
}

__global__ void slg_den_kernel(const float* x, const float* vc, const float* vs, float s, float scale, float* out,
                               std::int64_t n) {
  MI_GRID_LOOP(i, n) {
    const float dc = sub(x[i], mul(vc[i], s)), ds = sub(x[i], mul(vs[i], s));
    out[i] = add(out[i], mul(sub(dc, ds), scale));
  }
}

#undef MI_GRID_LOOP

}  // namespace

void compute_coefficients(CoefKind kind, const float* sigmas, int steps, float eta, float eta_plus_one, float* coef,
                          cudaStream_t stream) {
  coef_kernel<<<1, 1, 0, stream>>>(kind, sigmas, steps, eta, eta_plus_one, coef);
  MI_CUDA_CHECK_LAUNCH();
}

#define MI_LAUNCH(kernel, n, ...)                                     \
  do {                                                                \
    if ((n) > 0) {                                                    \
      kernel<<<blocks(n), kThreads, 0, stream>>>(__VA_ARGS__, (n));   \
      MI_CUDA_CHECK_LAUNCH();                                         \
    }                                                                 \
  } while (0)

void torch_expm1(const float* x, float* y, std::int64_t n, cudaStream_t stream) { MI_LAUNCH(expm1_kernel, n, x, y); }

void euler(float* x, const float* den, float s, float dt, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(euler_kernel, n, x, den, s, dt);
}
void velocity_euler(float* x, const float* v, float h, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(velocity_euler_kernel, n, x, v, h);
}
void ab2(float* x, const float* v, const float* v_prev, float h, float a, float b, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(ab2_kernel, n, x, v, v_prev, h, a, b);
}
void ancestral(float* x, const float* den, const float* noise, float ratio, float keep, float alpha, float renoise,
               float s_noise, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(ancestral_kernel, n, x, den, noise, ratio, keep, alpha, renoise, s_noise);
}
void heun_predict(const float* x, const float* den, float s, float dt, float* d, float* x2, std::int64_t n,
                  cudaStream_t stream) {
  MI_LAUNCH(heun_predict_kernel, n, x, den, s, dt, d, x2);
}
void heun_correct(float* x, const float* d, const float* x2, const float* den2, float s_next, float dt, std::int64_t n,
                  cudaStream_t stream) {
  MI_LAUNCH(heun_correct_kernel, n, x, d, x2, den2, s_next, dt);
}
void dpm2m(float* x, const float* den, const float* old, float c1, float c2, float a, float q, std::int64_t n,
           cudaStream_t stream) {
  MI_LAUNCH(dpm2m_kernel, n, x, den, old, c1, c2, a, q);
}
void dpm2m_sde(float* x, const float* den, const float* old, const float* noise, float c1, float c2, float c3,
               float s_next, float c_noise, float s_noise, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(dpm2m_sde_kernel, n, x, den, old, noise, c1, c2, c3, s_next, c_noise, s_noise);
}
void er_sde(float* x, const float* den, const float* old, float* dd, const float* old_dd, const float* noise, int stage,
            float c1, float c2, float s2, float div2, float s3, float div3, float alpha, float s_noise, float c_noise,
            std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(er_sde_kernel, n, x, den, old, dd, old_dd, noise, stage, c1, c2, s2, div2, s3, div3, alpha, s_noise, c_noise);
}
void res_multistep(float* x, const float* den, const float* old, float e, float h, float b1, float b2, std::int64_t n,
                   cudaStream_t stream) {
  MI_LAUNCH(res_kernel, n, x, den, old, e, h, b1, b2);
}
void dataspace_mix(const float* z, const float* x0, const float* q, int channels, std::int64_t pixels, float* x_in,
                   cudaStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(channels) * pixels;
  if (total > 0) {
    dataspace_mix_kernel<<<blocks(total), kThreads, 0, stream>>>(z, x0, q, channels, pixels, x_in);
    MI_CUDA_CHECK_LAUNCH();
  }
}
void dataspace_update(float* x0, const float* z, const float* v, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(sub_kernel, n, x0, z, v);
}
void cfg_velocity(const float* c, const float* u, float g, float* out, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(cfg_velocity_kernel, n, c, u, g, out);
}
void slg_velocity(const float* p, const float* c, const float* k, float s, float* out, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(slg_velocity_kernel, n, p, c, k, s, out);
}
void noise_scaling(float* noise, const float* latent, float sigma, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(noise_scaling_kernel, n, noise, latent, sigma);
}
void velocity_to_denoised(const float* x, const float* v, float sigma, float* out, std::int64_t n, cudaStream_t stream) {
  MI_LAUNCH(v2d_kernel, n, x, v, sigma, out);
}
void cfg_denoised(const float* x, const float* vc, const float* vu, float sigma, float g, float* out, std::int64_t n,
                  cudaStream_t stream) {
  MI_LAUNCH(cfg_den_kernel, n, x, vc, vu, sigma, g, out);
}
void slg_denoised(const float* x, const float* vc, const float* vs, float sigma, float scale, float* out, std::int64_t n,
                  cudaStream_t stream) {
  MI_LAUNCH(slg_den_kernel, n, x, vc, vs, sigma, scale, out);
}

#undef MI_LAUNCH

}  // namespace mi::pipeline::sampler_kernels
