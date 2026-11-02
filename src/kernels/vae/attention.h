// VAE mid-block attention (fp32, single head) for both FLUX.2 widths: head dim D = 384 (the small decoder) and
// D = 512 (the full decoder and the encoder). The sm_80+ kernel is templated on D (attention.cu); its D = 384
// instance is the original kernel bit for bit. Volta mid_attention uses bounded FP32 score slabs in caller scratch;
// the standalone no-workspace API keeps the compatibility kernel.
//
// The types and functions are declared in kernels/vae/vae_kernels.h together with the other VAE ops (their
// comments there predate D = 512); this header is the attention API's home and states the head-dim contract:
//
//   vae::attention_fp32(q, k, v, o, scale, precision, stream)
//       q, k, v: [B, L, D] fp32 views (row stride may exceed D, 16-byte aligned); o: contiguous [B, L, D], must not
//       overlap q/k/v; D = q.dim(2) in {384, 512}. Never materializes L x L.
//   vae::mid_attention_workspace_bytes(x)          x: [B, H, W, D]
//   vae::mid_attention(lt, x, weights, residual, y, workspace, workspace_bytes, precision, fast_linears, stream)
//       y = residual + to_out(SDPA(to_q/k/v(GN(x)))) for x, y [B, H, W, D]; weights [D, D] / [D] fp32 (torch layouts);
//       scale 1/sqrt(D) in fp32.
//   AttnPrecision::kTf32x3: fp32-accurate (the reference's SDPA), kTf32: one TF32 pass.
#pragma once

#include <cstdint>

#include "kernels/vae/vae_kernels.h"

namespace mi::vae {

// True for the head dims attention_fp32 / mid_attention accept (384, 512).
bool attention_head_dim_supported(std::int64_t d);

}  // namespace mi::vae
