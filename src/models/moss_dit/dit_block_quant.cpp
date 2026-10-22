#include "models/moss_dit/dit_block_quant.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/attention/flash_attention.h"
#include "kernels/attention_q8/attention_q8.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/norm.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/fused/qk_norm_rope.h"
#include "kernels/lora/lora_gemm.h"
#include "kernels/quant/quant.h"

namespace mi::moss_dit {
namespace {

namespace kq = kernels::quant;

constexpr std::int64_t kD = 3072, kH = 24, kHd = 128, kF = 8192;
static_assert(kFusedBigCols * 2 >= kF * 2 + kF, "the INT8 ffn_down input must fit next to the fused ffn output");
// SDPA's default scale 1/sqrt(head_dim), computed in double and rounded once as torch does (as moss_dit.cpp).
const float kScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHd)));
attention_q8::Variant variant_of(AttentionMode m) {
  switch (m) {
    case AttentionMode::kQ8Fp16: return attention_q8::Variant::kInt8Fp16;
    case AttentionMode::kQ8Fp16F32Acc: return attention_q8::Variant::kInt8Fp16F32Acc;
    case AttentionMode::kQ8Fp8F32Acc: return attention_q8::Variant::kInt8Fp8F32Acc;
    default: return attention_q8::Variant::kInt8Fp8;
  }
}

attention_q8::Options q8_options(AttentionMode m) {
  attention_q8::Options o;
  o.variant = variant_of(m);
  return o;
}

constexpr AttentionMode kModes[] = {AttentionMode::kExact, AttentionMode::kQ8Fp16, AttentionMode::kQ8Fp16F32Acc,
                                    AttentionMode::kQ8Fp8, AttentionMode::kQ8Fp8F32Acc};

Tensor rows_of(const Tensor& buf, std::int64_t offset, std::int64_t rows, std::int64_t cols, std::int64_t t0,
               std::int64_t t1) {
  return Tensor::view(static_cast<std::uint16_t*>(buf.data) + offset, DType::kBF16, Device::kCUDA, {rows, cols})
      .slice0(t0, t1);
}

Tensor heads(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, Device::kCUDA, {buf.dim(0), kH, kHd});
  t.stride[0] = buf.stride[0];
  return t;
}

Tensor vec(const Tensor& mod, std::int64_t i) {
  return Tensor::view(static_cast<std::uint16_t*>(mod.data) + i * kD, DType::kBF16, Device::kCUDA, {kD});
}

const __nv_bfloat16* bf(const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); }
__nv_bfloat16* bf_mut(const Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); }

bool aligned256(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 256 == 0; }

kq::Int8Operands operands(const std::int8_t* xq, std::int64_t ldx, const float* scale, const Int8Linear& w,
                          std::int64_t m) {
  kq::Int8Operands op;
  op.x_q = xq;
  op.ldx = ldx;
  op.x_scale = scale;
  op.w = static_cast<const std::int8_t*>(w.w.data);
  op.w_scale = static_cast<const float*>(w.scale.data);
  op.m = m;
  op.n = w.w.dim(0);
  op.k = w.w.dim(1);
  return op;
}

bool bf16_up_gate_runs_op_by_op(const BlockTensors& w, const FusedRoute& route, bool lora) {
  return w.up_gate8.empty() && !route.up_gate && !lora;
}

}  // namespace

const char* attention_mode_name(AttentionMode m) {
  switch (m) {
    case AttentionMode::kExact: return "exact";
    case AttentionMode::kQ8Fp16: return "q8_fp16";
    case AttentionMode::kQ8Fp16F32Acc: return "q8_fp16_f32acc";
    case AttentionMode::kQ8Fp8: return "q8_fp8";
    case AttentionMode::kQ8Fp8F32Acc: return "q8_fp8_f32acc";
  }
  return "?";
}

AttentionMode parse_attention_mode(std::string_view name) {
  if (name == "q8") return AttentionMode::kQ8Fp8;
  for (const AttentionMode m : kModes) {
    if (name == attention_mode_name(m)) return m;
  }
  throw std::invalid_argument("attention must be exact, q8 (= q8_fp8), q8_fp16, q8_fp16_f32acc, q8_fp8 or q8_fp8_f32acc, got \"" +
                              std::string(name) + "\"");
}

bool attention_supported(AttentionMode m) { return m == AttentionMode::kExact || attention_q8::supported(variant_of(m)); }

std::int64_t quant_big_cols(std::int64_t hw, std::span<const std::int64_t> text_lengths, bool bf16_up_gate,
                            const FusedRouteSnapshot& route) {
  return bf16_up_gate ? fused_big_cols(hw, text_lengths, route) : kFusedBigCols;
}

std::size_t q8_attention_workspace_bytes(std::int64_t hw, std::span<const std::int64_t> text_lengths, AttentionMode mode) {
  if (mode == AttentionMode::kExact) return 0;
  const std::size_t n = text_lengths.size();
  if (n > 16) throw std::invalid_argument("mi::moss_dit::q8_attention_workspace_bytes: too many rows");
  std::size_t bytes = 0;
  // Every subset of the rows: a run is contiguous rows of a forward's row set, and callers size the workspace for the
  // largest row set (a step without SLG runs a subset of it).
  for (std::uint32_t mask = 1; mask < (std::uint32_t{1} << n); ++mask) {
    std::int64_t total = 0, max_len = 0;
    int batch = 0;
    for (std::size_t r = 0; r < n; ++r) {
      if (!(mask & (std::uint32_t{1} << r))) continue;
      const std::int64_t len = hw + text_lengths[r];
      total += len;
      max_len = std::max(max_len, len);
      ++batch;
    }
    bytes = std::max(bytes, attention_q8::attention_q8_workspace_bytes(q8_options(mode), total, total, batch,
                                                                       static_cast<int>(max_len), kH, kH));
  }
  return bytes;
}

void run_block_quant(const BlockTensors& w, const QuantBlockBuffers& qa, const BlockRun& run, const QkvGemm& qkv_gemm,
                     const FusedRouteSnapshot& snapshot, AttentionMode attention, cudaStream_t stream, int block) {
  namespace kl = kernels::lora;
  const FusedBlockBuffers& a = qa.f;
  const std::int64_t l = a.x.dim(0), m = run.t1 - run.t0;
  const FusedRoute route = fused_route(m, snapshot.validated);
  const auto site = [&](lora::Site s) { return qa.lora != nullptr ? qa.lora->site(block, s) : nullptr; };
  const lora::SiteBranch* const s_qkv = site(lora::Site::kQkv);
  const lora::SiteBranch* const s_o = site(lora::Site::kO);
  const lora::SiteBranch* const s_up = site(lora::Site::kUpGate);
  const lora::SiteBranch* const s_down = site(lora::Site::kDown);
  if (s_qkv != nullptr && w.qkv_split) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: a LoRA on the QKV of a block whose q, k and v differ in "
                                "precision is not supported");
  }
  if (qa.lora != nullptr && qa.lora->any_active(block) &&
      (qa.lora_t.dtype != DType::kBF16 || qa.lora_t.rank != 2 || qa.lora_t.dim(0) < l ||
       qa.lora_t.dim(1) < qa.lora->max_rank() || qa.lora_workspace_bytes < lora::t_workspace_bytes(l, qa.lora->max_rank()))) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: LoRA buffers do not match the side branches");
  }
  // t of a site for its bf16 input `in` (rows of this run): the LoRA GEMM's operand.
  const auto prepare = [&](const lora::SiteBranch& sb, const Tensor& in) {
    Tensor t = Tensor::view(qa.lora_t.data, DType::kBF16, Device::kCUDA, {m, sb.rank});
    t.stride[0] = qa.lora_t.stride[0];
    lora::compute_t(sb, in, t, qa.lora_workspace, qa.lora_workspace_bytes, *qkv_gemm.gemm, qkv_gemm.workspace,
                    qkv_gemm.workspace_bytes, stream);
    return lora::lora_slices(sb, t);
  };
  const bool up_gate_op_by_op = bf16_up_gate_runs_op_by_op(w, route, s_up != nullptr);
  const std::int64_t big_cols = up_gate_op_by_op ? kOpByOpBigCols : kFusedBigCols;
  std::size_t attn_need = 0;
  if (attention != AttentionMode::kExact) {
    const int batch = static_cast<int>(run.cu_seqlens.numel()) - 1;
    attn_need = attention_q8::attention_q8_workspace_bytes(q8_options(attention), m, m, batch, run.max_seqlen, kH, kH);
  }
  if (qkv_gemm.gemm == nullptr || !a.big.is_contiguous() || a.big.numel() < l * big_cols || a.mod.numel() != 6 * kD ||
      qa.scale.dtype != DType::kF32 || qa.scale.numel() < l || qa.attn_workspace_bytes < attn_need ||
      (attn_need > 0 && qa.attn_workspace == nullptr)) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: buffers do not match x " + a.x.describe() + ": big " +
                                a.big.describe() + " (" + std::to_string(big_cols) + " per token), scale " +
                                qa.scale.describe() + ", attention workspace " + std::to_string(qa.attn_workspace_bytes) +
                                " of " + std::to_string(attn_need) + " bytes");
  }
  for (const void* p : {a.x.data, a.y.data, a.big.data}) {
    if (!aligned256(p)) throw std::invalid_argument("mi::moss_dit::run_block_quant: operands must be 256-byte aligned");
  }

  const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1);
  const Tensor qkv = rows_of(a.big, 0, l, 3 * kD, run.t0, run.t1);
  const Tensor attn = rows_of(a.big, l * 3 * kD, l, kD, run.t0, run.t1);
  float* scale = static_cast<float*>(qa.scale.data) + run.t0;
  // INT8 [L, kD] GEMM inputs: the qkv input in the attention output's columns (free until attention runs), the o-proj
  // and ffn_up_gate inputs in y (free while their GEMMs run: the INT8 path never stores the bf16 norm output).
  std::int8_t* const xq_qkv = static_cast<std::int8_t*>(attn.data);
  std::int8_t* const xq_y = static_cast<std::int8_t*>(a.y.data) + run.t0 * kD;
  const auto linear = [&](const Tensor& out, const Tensor& in, const Tensor& weight) {
    qkv_gemm.gemm->linear(out, in, weight, nullptr, qkv_gemm.workspace, qkv_gemm.workspace_bytes, stream);
  };

  // QKV projection (P:139-140 up to the linear).
  if (!w.qkv8.empty() && s_qkv != nullptr) {
    kernels::rms_norm_modulate_moss(x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), y, stream);
    kq::rotate_quantize(bf(y), kD, xq_qkv, kD, scale, m, static_cast<int>(kD), stream);
    kl::int8_gemm_lora(operands(xq_qkv, kD, scale, w.qkv8, m), nullptr, prepare(*s_qkv, y), bf_mut(qkv), 3 * kD, stream);
  } else if (!w.qkv8.empty()) {
    kq::rms_norm_modulate_rotate_quantize(bf(x), kD, bf(w.norm1), bf(vec(a.mod, 1)), bf(vec(a.mod, 0)), xq_qkv, kD, scale,
                                          m, static_cast<int>(kD), stream);
    kq::int8_gemm(operands(xq_qkv, kD, scale, w.qkv8, m), nullptr, bf_mut(qkv), 3 * kD, stream);
  } else {
    kernels::rms_norm_modulate_moss(x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), y, stream);
    if (!w.qkv_split && s_qkv != nullptr) {
      kl::gemm_lora(qkv, y, w.qkv, prepare(*s_qkv, y), stream);
    } else if (!w.qkv_split) {
      linear(qkv, y, w.qkv);
    } else {
      // q, k and v of different precisions: one shared quantized input for the INT8 parts, the bf16 norm output for
      // the others, each GEMM into its columns of qkv.
      bool any_int8 = false;
      for (const Int8Linear& p : w.qkv8_part) any_int8 = any_int8 || !p.empty();
      if (any_int8) kq::rotate_quantize(bf(y), kD, xq_qkv, kD, scale, m, static_cast<int>(kD), stream);
      for (int i = 0; i < 3; ++i) {
        Tensor out = Tensor::view(static_cast<std::uint16_t*>(qkv.data) + i * kD, DType::kBF16, Device::kCUDA, {m, kD});
        out.stride[0] = 3 * kD;
        if (!w.qkv8_part[i].empty()) {
          kq::int8_gemm(operands(xq_qkv, kD, scale, w.qkv8_part[i], m), nullptr, bf_mut(out), 3 * kD, stream);
        } else {
          linear(out, y, w.qkv_part[i]);
        }
      }
    }
  }
  kernels::fused::qk_norm_rope_moss(qkv, static_cast<int>(kH), w.q_norm, w.k_norm, a.rope.slice0(run.t0, run.t1), stream);
  if (attention != AttentionMode::kExact) {
    attention_q8::attention_q8(q8_options(attention), heads(qkv, 0), heads(qkv, kD), heads(qkv, 2 * kD), heads(attn, 0),
                               run.cu_seqlens, run.cu_seqlens, run.max_seqlen, kScale, qa.attn_workspace,
                               qa.attn_workspace_bytes, stream);
  } else {
    attention::flash_fwd(heads(qkv, 0), heads(qkv, kD), heads(qkv, 2 * kD), heads(attn, 0), run.cu_seqlens,
                         run.cu_seqlens, run.max_seqlen, false, kScale, a.lse, a.lse_bytes, stream);
  }

  // x += o(attn) * gate_a (P:142).
  if (!w.o8.empty()) {
    kl::LoraSlices lo;
    if (s_o != nullptr) lo = prepare(*s_o, attn);  // t from the bf16 input, before it is rotated
    kq::rotate_quantize(bf(attn), kD, xq_y, kD, scale, m, static_cast<int>(kD), stream);
    if (s_o != nullptr) {
      kl::int8_gemm_gated_residual_lora(operands(xq_y, kD, scale, w.o8, m), nullptr, bf(vec(a.mod, 2)), lo, bf_mut(x), kD,
                                        stream);
    } else {
      kq::int8_gemm_gated_residual(operands(xq_y, kD, scale, w.o8, m), nullptr, bf(vec(a.mod, 2)), bf_mut(x), kD, stream);
    }
  } else if (s_o != nullptr) {
    kl::gemm_gated_residual_lora(x, attn, w.o, vec(a.mod, 2), prepare(*s_o, attn), stream);
  } else if (route.o) {
    kernels::fused::gemm_gated_residual(x, attn, w.o, vec(a.mod, 2), stream);
  } else {
    linear(y, attn, w.o);
    kernels::gated_residual(x, y, vec(a.mod, 2), x, stream);
  }

  // ffn = situ_glu(up_gate(rms_norm_modulate(x))) (P:143-147).
  Tensor ffn;
  if (!w.up_gate8.empty() && s_up != nullptr) {
    // The bf16 norm output stays in y: the quantized input goes to the attention output's columns (free now, and
    // after the [L, kF] ffn this GEMM writes).
    ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
    kernels::rms_norm_modulate_moss(x, w.norm2, vec(a.mod, 4), vec(a.mod, 3), y, stream);
    kq::rotate_quantize(bf(y), kD, xq_qkv, kD, scale, m, static_cast<int>(kD), stream);
    kl::int8_gemm_situ_glu_lora(operands(xq_qkv, kD, scale, w.up_gate8, m), prepare(*s_up, y), bf_mut(ffn), kF, stream);
  } else if (!w.up_gate8.empty()) {
    ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
    kq::rms_norm_modulate_rotate_quantize(bf(x), kD, bf(w.norm2), bf(vec(a.mod, 4)), bf(vec(a.mod, 3)), xq_y, kD, scale,
                                          m, static_cast<int>(kD), stream);
    kq::int8_gemm_situ_glu(operands(xq_y, kD, scale, w.up_gate8, m), bf_mut(ffn), kF, stream);
  } else {
    kernels::rms_norm_modulate_moss(x, w.norm2, vec(a.mod, 4), vec(a.mod, 3), y, stream);
    if (s_up != nullptr) {
      ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
      kl::gemm_situ_glu_lora(ffn, y, w.up_gate, prepare(*s_up, y), stream);
    } else if (!up_gate_op_by_op) {
      ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
      kernels::fused::gemm_situ_glu(ffn, y, w.up_gate, stream);
    } else {
      const Tensor up_gate = rows_of(a.big, 0, l, 2 * kF, run.t0, run.t1);
      ffn = rows_of(a.big, l * 2 * kF, l, kF, run.t0, run.t1);
      linear(up_gate, y, w.up_gate);
      kernels::situ_glu(up_gate, ffn, stream);
    }
  }

  // x += down(ffn) * gate_f (P:148).
  if (!w.down8.empty()) {
    // [L, kF] int8 where the FFN leaves `big` free: after the fused [L, kF] ffn, or over the op-by-op up_gate output.
    std::int8_t* const base =
        up_gate_op_by_op ? static_cast<std::int8_t*>(a.big.data) : static_cast<std::int8_t*>(a.big.data) + l * kF * 2;
    std::int8_t* const xq = base + run.t0 * kF;
    kl::LoraSlices ld;
    if (s_down != nullptr) ld = prepare(*s_down, ffn);
    kq::rotate_quantize(bf(ffn), kF, xq, kF, scale, m, static_cast<int>(kF), stream);
    if (s_down != nullptr) {
      kl::int8_gemm_gated_residual_lora(operands(xq, kF, scale, w.down8, m), nullptr, bf(vec(a.mod, 5)), ld, bf_mut(x), kD,
                                        stream);
    } else {
      kq::int8_gemm_gated_residual(operands(xq, kF, scale, w.down8, m), nullptr, bf(vec(a.mod, 5)), bf_mut(x), kD, stream);
    }
  } else if (s_down != nullptr) {
    kl::gemm_gated_residual_lora(x, ffn, w.down, vec(a.mod, 5), prepare(*s_down, ffn), stream);
  } else if (route.down) {
    kernels::fused::gemm_gated_residual(x, ffn, w.down, vec(a.mod, 5), stream);
  } else {
    linear(y, ffn, w.down);
    kernels::gated_residual(x, y, vec(a.mod, 5), x, stream);
  }
}

}  // namespace mi::moss_dit
