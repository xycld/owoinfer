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
#include "kernels/nvfp4/nvfp4.h"
#include "kernels/quant/quant.h"
#include "kernels/quant4/quant4.h"
#include "models/moss_dit/dit_kernels.h"

namespace mi::moss_dit {
namespace {

namespace kq = kernels::quant;
namespace kq4 = kernels::quant4;
namespace kn = kernels::nvfp4;

constexpr std::int64_t kD = 3072, kH = 24, kHd = 128, kF = 8192;
static_assert(kFusedBigCols * 2 >= kF * 2 + kF, "the INT8 ffn_down input must fit next to the fused ffn output");
static_assert(kInt4UnpackBytes >= 2 * kF * kD && kInt4UnpackBytes >= kD * kF, "every INT4 block weight fits unpacked");
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

kq4::Int4Operands operands4(const std::uint8_t* xq, std::int64_t ldx, const float* scale, const Int4Linear& w,
                            std::int64_t m) {
  kq4::Int4Operands op;
  op.x_q = xq;
  op.ldx = ldx;
  op.x_scale = scale;
  op.w = static_cast<const std::uint8_t*>(w.w.data);
  op.w_scale = static_cast<const float*>(w.scale.data);
  op.m = m;
  op.n = w.w.dim(0);
  op.k = w.k();
  return op;
}

kn::Nvfp4Operands operands_nv(const kn::Nvfp4Act& x, const Nvfp4Linear& w, std::int64_t m) {
  kn::Nvfp4Operands op;
  op.x = x.q;
  op.x_sf = x.sf;
  op.x_scale = x.scale;
  op.w = static_cast<const std::uint8_t*>(w.w.data);
  op.w_sf = static_cast<const std::uint8_t*>(w.sf.data);
  op.w_scale2 = w.scale2;
  op.m = m;
  op.n = w.w.dim(0);
  op.k = w.k();
  return op;
}

bool bf16_up_gate_runs_op_by_op(const BlockTensors& w, const FusedRoute& route, bool lora) {
  return w.up_gate8.empty() && w.up_gate4.empty() && w.up_gate_nvfp4.empty() && !route.up_gate && !lora;
}

}  // namespace

bool int4_mma_native(int major, int minor) {
  static_cast<void>(minor);
  return major == 8;
}

bool int4_mma_native() {
  int device = 0, major = 0, minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return true;
  }
  return int4_mma_native(major, minor);
}

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
                            const FusedRouteSnapshot& route, std::int64_t nvfp4_cols) {
  return std::max(bf16_up_gate ? fused_big_cols(hw, text_lengths, route) : kFusedBigCols, nvfp4_cols);
}

std::int64_t nvfp4_big_cols(bool nvfp4_up_gate, bool bf16_ffn) {
  return !nvfp4_up_gate ? 0 : bf16_ffn ? kOpByOpBigCols : 2 * kF;
}

std::size_t nvfp4_sf_bytes(std::int64_t l) { return static_cast<std::size_t>(kn::sf_bytes(l, kF)); }

std::size_t q8_attention_workspace_bytes(std::int64_t hw, std::span<const std::int64_t> text_lengths, AttentionMode mode,
                                         std::int64_t ref_tokens) {
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
    bytes = std::max(bytes, attention_q8::attention_q8_workspace_bytes(q8_options(mode), total, total + batch * ref_tokens,
                                                                       batch, static_cast<int>(max_len + ref_tokens), kH, kH));
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
  // INT4 sites that run the INT8 GEMMs on unpacked operands: every INT4 site on a GPU without INT4 MMA
  // (qa.int4_via_int8), else those with a LoRA side branch.
  const bool via = qa.int4_via_int8;
  const bool u_qkv = via || s_qkv != nullptr, u_o = via || s_o != nullptr, u_up = via || s_up != nullptr,
             u_down = via || s_down != nullptr;
  const bool unpack = (u_qkv && (!w.qkv4.empty() || !w.qkv4_part[0].empty() || !w.qkv4_part[1].empty() ||
                                 !w.qkv4_part[2].empty())) ||
                      (u_o && !w.o4.empty()) || (u_up && !w.up_gate4.empty()) || (u_down && !w.down4.empty());
  if (unpack && (qa.int4_unpack == nullptr || qa.int4_unpack_bytes < kInt4UnpackBytes)) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: an INT4 linear run through the INT8 GEMMs (LoRA side "
                                "branch, or int4_via_int8) needs the int4_unpack buffer (kInt4UnpackBytes)");
  }
  // t of a site for its bf16 input `in` (rows of this run): the LoRA GEMM's operand.
  const auto prepare = [&](const lora::SiteBranch& sb, const Tensor& in) {
    Tensor t = Tensor::view(qa.lora_t.data, DType::kBF16, Device::kCUDA, {m, sb.rank});
    t.stride[0] = qa.lora_t.stride[0];
    lora::compute_t(sb, in, t, qa.lora_workspace, qa.lora_workspace_bytes, *qkv_gemm.gemm, qkv_gemm.workspace,
                    qkv_gemm.workspace_bytes, stream);
    return lora::lora_slices(sb, t);
  };
  // An NVFP4 ffn_up_gate writes its whole [m, 2F] output (no SiTU epilogue): the ffn_down quantizer applies SiTU-GLU as
  // it reads it when ffn_down is NVFP4 without a side branch; otherwise the bf16 ffn is materialized after it, in the
  // op-by-op layout.
  const bool nv_up = !w.up_gate_nvfp4.empty();
  const bool nv_ffn_fused = nv_up && !w.down_nvfp4.empty() && s_down == nullptr;
  const bool up_gate_op_by_op = bf16_up_gate_runs_op_by_op(w, route, s_up != nullptr);
  // With a side branch on an NVFP4 ffn_up_gate its codes go after the [L, 2F] output, where the op-by-op layout keeps
  // the ffn (written only after the GEMM): the 3F columns too.
  const std::int64_t big_cols =
      up_gate_op_by_op ? kOpByOpBigCols
                       : std::max(kFusedBigCols, nvfp4_big_cols(nv_up, !nv_ffn_fused || s_up != nullptr));
  // Columns of Δ this block needs: its widest NVFP4 site with a side branch.
  std::int64_t delta_cols = 0;
  if (s_qkv != nullptr && !w.qkv_nvfp4.empty()) delta_cols = std::max(delta_cols, 3 * kD);
  if (s_o != nullptr && !w.o_nvfp4.empty()) delta_cols = std::max(delta_cols, kD);
  if (s_up != nullptr && nv_up) delta_cols = std::max(delta_cols, 2 * kF);
  if (s_down != nullptr && !w.down_nvfp4.empty()) delta_cols = std::max(delta_cols, kD);
  if (w.nvfp4() && (qa.nvfp4_sf == nullptr || qa.nvfp4_sf_bytes < nvfp4_sf_bytes(l))) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: NVFP4 linears need the nvfp4_sf buffer (nvfp4_sf_bytes)");
  }
  if (delta_cols > 0 &&
      (qa.nvfp4_delta.dtype != DType::kBF16 || qa.nvfp4_delta.numel() < l * delta_cols || qa.nvfp4_scratch == nullptr ||
       qa.nvfp4_scratch_bytes < lora::delta_scratch_bytes(l, qa.lora->max_rank(), delta_cols))) {
    throw std::invalid_argument("mi::moss_dit::run_block_quant: a LoRA side branch on an NVFP4 linear needs the nvfp4_delta "
                                "(" + std::to_string(delta_cols) + " columns per token) and nvfp4_scratch buffers");
  }
  std::size_t attn_need = 0;
  if (attention != AttentionMode::kExact) {
    const int batch = static_cast<int>(run.cu_seqlens.numel()) - 1;
    const std::int64_t keys = run.slot_rows == 0 ? m : run.k1 - run.k0;
    attn_need = attention_q8::attention_q8_workspace_bytes(q8_options(attention), m, keys, batch, run.max_seqlen, kH, kH);
  }
  if (qkv_gemm.gemm == nullptr || !a.big.is_contiguous() ||
      a.big.numel() < std::max(l * big_cols, slotted_big(l, run.slot_rows)) || a.mod.numel() != 6 * kD ||
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
  const std::int64_t slots = run.slot_rows;
  const Tensor qkv = rows_of(a.big, slots * 3 * kD, l, 3 * kD, run.t0, run.t1);
  const Tensor attn = rows_of(a.big, (l + 2 * slots) * 3 * kD, l, kD, run.t0, run.t1);
  // The keys and values: the run's own rows, or with reference slots the run's key rows of the slotted region.
  const Tensor kv = slots == 0 ? qkv : rows_of(a.big, 0, l + 2 * slots, 3 * kD, run.k0, run.k1);
  float* scale = static_cast<float*>(qa.scale.data) + run.t0;
  // Quantized [L, kD] GEMM inputs (INT8, or packed INT4 in the first half): the qkv input in the attention output's
  // columns (free until attention runs), the o-proj and ffn_up_gate inputs in y (free while their GEMMs run: the
  // quantized paths never store the bf16 norm output). An INT4 linear with a LoRA side branch runs the INT8 LoRA GEMM
  // on its values unpacked to INT8 (the same int32 dot products, so the base result is the INT4 GEMM's): the packed
  // input then sits right after its unpacked INT8 copy (each region holds 2 kD bytes per token, the pair needs 1.5 kD).
  std::int8_t* const xq_qkv = static_cast<std::int8_t*>(attn.data);
  std::int8_t* const xq_y = static_cast<std::int8_t*>(a.y.data) + run.t0 * kD;
  const auto linear = [&](const Tensor& out, const Tensor& in, const Tensor& weight) {
    qkv_gemm.gemm->linear(out, in, weight, nullptr, qkv_gemm.workspace, qkv_gemm.workspace_bytes, stream);
  };
  const auto u8 = [](std::int8_t* p) { return reinterpret_cast<std::uint8_t*>(p); };
  // INT8 operands of an INT4 linear for the INT8 GEMMs: its packed input `xq4` (row stride k / 2) unpacked to `xq8`
  // (row stride k) and its weight unpacked into the int4_unpack buffer.
  const auto unpacked = [&](const std::int8_t* xq4, std::int8_t* xq8, const Int4Linear& lin) {
    const std::int64_t k = lin.k(), n = lin.w.dim(0);
    unpack_int4(xq4, k / 2, xq8, k, m, k, stream);
    unpack_int4(lin.w.data, k / 2, qa.int4_unpack, k, n, k, stream);
    kq::Int8Operands op;
    op.x_q = xq8;
    op.ldx = k;
    op.x_scale = scale;
    op.w = static_cast<const std::int8_t*>(qa.int4_unpack);
    op.w_scale = static_cast<const float*>(lin.scale.data);
    op.m = m;
    op.n = n;
    op.k = k;
    return op;
  };

  // NVFP4 inputs: the codes at `q` (a run-local [m, k / 2] region), the swizzled scales in nvfp4_sf and the per-token
  // scales at the start of `scale`'s buffer (16-byte aligned; the runs of a block are sequential on one stream).
  const auto nv_act = [&](std::int8_t* q) {
    return kn::Nvfp4Act{u8(q), static_cast<std::uint8_t*>(qa.nvfp4_sf), static_cast<float*>(qa.scale.data)};
  };
  // The E1L / E2L term of an NVFP4 site with a side branch, from its bf16 input `in` (rows of this run): t, then
  // Δ = t · Bᵀ stored in bf16 (natural: ffn_up_gate's stored column order).
  const auto nv_delta = [&](const lora::SiteBranch& sb, const Tensor& in, bool natural) {
    Tensor t = Tensor::view(qa.lora_t.data, DType::kBF16, Device::kCUDA, {m, sb.rank});
    t.stride[0] = qa.lora_t.stride[0];
    lora::compute_t(sb, in, t, qa.lora_workspace, qa.lora_workspace_bytes, *qkv_gemm.gemm, qkv_gemm.workspace,
                    qkv_gemm.workspace_bytes, stream);
    const Tensor d = Tensor::view(qa.nvfp4_delta.data, DType::kBF16, Device::kCUDA, {m, sb.n});
    lora::compute_delta(sb, t, d, natural, qa.nvfp4_scratch, qa.nvfp4_scratch_bytes, *qkv_gemm.gemm, qkv_gemm.workspace,
                        qkv_gemm.workspace_bytes, stream);
    return kn::Nvfp4Delta{bf(d), sb.n};
  };

  // QKV projection (P:139-140 up to the linear).
  if (!w.qkv_nvfp4.empty() && s_qkv != nullptr) {
    kernels::rms_norm_modulate_moss(x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), y, stream);
    const kn::Nvfp4Delta d = nv_delta(*s_qkv, y, false);
    const kn::Nvfp4Act act = nv_act(xq_qkv);
    kn::quantize_nvfp4(bf(y), kD, act, m, static_cast<int>(kD), stream);
    kn::nvfp4_gemm(operands_nv(act, w.qkv_nvfp4, m), nullptr, bf_mut(qkv), 3 * kD, stream, kn::Nvfp4Tile::kAuto, d);
  } else if (!w.qkv_nvfp4.empty()) {
    const kn::Nvfp4Act act = nv_act(xq_qkv);
    kn::rms_norm_modulate_quantize_nvfp4(bf(x), kD, bf(w.norm1), bf(vec(a.mod, 1)), bf(vec(a.mod, 0)), act, m,
                                         static_cast<int>(kD), stream);
    kn::nvfp4_gemm(operands_nv(act, w.qkv_nvfp4, m), nullptr, bf_mut(qkv), 3 * kD, stream);
  } else if (!w.qkv4.empty() && s_qkv != nullptr) {
    kernels::rms_norm_modulate_moss(x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), y, stream);
    kq4::convrot_quantize_int4(bf(y), kD, u8(xq_qkv + m * kD), kD / 2, scale, m, static_cast<int>(kD), stream);
    const kq::Int8Operands op = unpacked(xq_qkv + m * kD, xq_qkv, w.qkv4);
    kl::int8_gemm_lora(op, nullptr, prepare(*s_qkv, y), bf_mut(qkv), 3 * kD, stream);
  } else if (!w.qkv4.empty() && via) {
    kq4::rms_norm_modulate_convrot_quantize_int4(bf(x), kD, bf(w.norm1), bf(vec(a.mod, 1)), bf(vec(a.mod, 0)),
                                                 u8(xq_qkv + m * kD), kD / 2, scale, m, static_cast<int>(kD), stream);
    kq::int8_gemm(unpacked(xq_qkv + m * kD, xq_qkv, w.qkv4), nullptr, bf_mut(qkv), 3 * kD, stream);
  } else if (!w.qkv4.empty()) {
    kq4::rms_norm_modulate_convrot_quantize_int4(bf(x), kD, bf(w.norm1), bf(vec(a.mod, 1)), bf(vec(a.mod, 0)), u8(xq_qkv),
                                                 kD / 2, scale, m, static_cast<int>(kD), stream);
    kq4::int4_gemm(operands4(u8(xq_qkv), kD / 2, scale, w.qkv4, m), nullptr, bf_mut(qkv), 3 * kD, stream);
  } else if (!w.qkv8.empty() && s_qkv != nullptr) {
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
      // q, k and v of different precisions (or NVFP4 parts with different tensor scales), each GEMM into its columns
      // of qkv: the bf16 parts from the norm output, the INT8 parts from one shared INT8 input, then the INT4 parts from
      // one shared INT4 input, then the NVFP4 parts from one shared NVFP4 input (each overwrites the earlier inputs and
      // scales once their GEMMs are queued: one stream).
      const auto part = [&](int i) {
        Tensor out = Tensor::view(static_cast<std::uint16_t*>(qkv.data) + i * kD, DType::kBF16, Device::kCUDA, {m, kD});
        out.stride[0] = 3 * kD;
        return out;
      };
      bool any_int8 = false, any_int4 = false, any_nvfp4 = false;
      for (int i = 0; i < 3; ++i) {
        any_int8 = any_int8 || !w.qkv8_part[i].empty();
        any_int4 = any_int4 || !w.qkv4_part[i].empty();
        any_nvfp4 = any_nvfp4 || !w.qkv_nvfp4_part[i].empty();
      }
      for (int i = 0; i < 3; ++i) {
        if (w.qkv8_part[i].empty() && w.qkv4_part[i].empty() && w.qkv_nvfp4_part[i].empty()) linear(part(i), y, w.qkv_part[i]);
      }
      if (any_int8) {
        kq::rotate_quantize(bf(y), kD, xq_qkv, kD, scale, m, static_cast<int>(kD), stream);
        for (int i = 0; i < 3; ++i) {
          if (!w.qkv8_part[i].empty()) {
            kq::int8_gemm(operands(xq_qkv, kD, scale, w.qkv8_part[i], m), nullptr, bf_mut(part(i)), 3 * kD, stream);
          }
        }
      }
      if (any_int4) {
        std::int8_t* const xq4 = via ? xq_qkv + m * kD : xq_qkv;
        kq4::convrot_quantize_int4(bf(y), kD, u8(xq4), kD / 2, scale, m, static_cast<int>(kD), stream);
        for (int i = 0; i < 3; ++i) {
          if (w.qkv4_part[i].empty()) continue;
          if (via) {
            kq::int8_gemm(unpacked(xq4, xq_qkv, w.qkv4_part[i]), nullptr, bf_mut(part(i)), 3 * kD, stream);
          } else {
            kq4::int4_gemm(operands4(u8(xq4), kD / 2, scale, w.qkv4_part[i], m), nullptr, bf_mut(part(i)), 3 * kD, stream);
          }
        }
      }
      if (any_nvfp4) {
        const kn::Nvfp4Act act = nv_act(xq_qkv);
        kn::quantize_nvfp4(bf(y), kD, act, m, static_cast<int>(kD), stream);
        for (int i = 0; i < 3; ++i) {
          if (!w.qkv_nvfp4_part[i].empty()) {
            kn::nvfp4_gemm(operands_nv(act, w.qkv_nvfp4_part[i], m), nullptr, bf_mut(part(i)), 3 * kD, stream);
          }
        }
      }
    }
  }
  kernels::fused::qk_norm_rope_moss(qkv, static_cast<int>(kH), w.q_norm, w.k_norm, a.rope.slice0(run.t0, run.t1), stream);
  if (run.kv_tap != nullptr) {
    tap_reference_kv(qkv, run.kv_tap, stream);
    if (run.tap_only) return;
  }
  const Tensor& cu_k = slots == 0 ? run.cu_seqlens : run.cu_seqlens_k;
  if (attention != AttentionMode::kExact) {
    attention_q8::attention_q8(q8_options(attention), heads(qkv, 0), heads(kv, kD), heads(kv, 2 * kD), heads(attn, 0),
                               run.cu_seqlens, cu_k, run.max_seqlen, kScale, qa.attn_workspace, qa.attn_workspace_bytes,
                               stream);
  } else {
    attention::flash_fwd(heads(qkv, 0), heads(kv, kD), heads(kv, 2 * kD), heads(attn, 0), run.cu_seqlens, cu_k,
                         run.max_seqlen, false, kScale, a.lse, a.lse_bytes, stream);
  }

  // x += o(attn) * gate_a (P:142).
  if (!w.o_nvfp4.empty()) {
    kn::Nvfp4Delta d;
    if (s_o != nullptr) d = nv_delta(*s_o, attn, false);  // from the bf16 input, before it is quantized
    const kn::Nvfp4Act act = nv_act(xq_y);
    kn::quantize_nvfp4(bf(attn), kD, act, m, static_cast<int>(kD), stream);
    kn::nvfp4_gemm_gated_residual(operands_nv(act, w.o_nvfp4, m), nullptr, bf(vec(a.mod, 2)), bf_mut(x), kD, stream,
                                  kn::Nvfp4Tile::kAuto, d);
  } else if (!w.o4.empty()) {
    kl::LoraSlices lo;
    if (s_o != nullptr) lo = prepare(*s_o, attn);  // t from the bf16 input, before it is rotated
    std::int8_t* const xq4 = u_o ? xq_y + m * kD : xq_y;
    kq4::convrot_quantize_int4(bf(attn), kD, u8(xq4), kD / 2, scale, m, static_cast<int>(kD), stream);
    if (s_o != nullptr) {
      kl::int8_gemm_gated_residual_lora(unpacked(xq4, xq_y, w.o4), nullptr, bf(vec(a.mod, 2)), lo, bf_mut(x), kD, stream);
    } else if (via) {
      kq::int8_gemm_gated_residual(unpacked(xq4, xq_y, w.o4), nullptr, bf(vec(a.mod, 2)), bf_mut(x), kD, stream);
    } else {
      kq4::int4_gemm_gated_residual(operands4(u8(xq4), kD / 2, scale, w.o4, m), nullptr, bf(vec(a.mod, 2)), bf_mut(x), kD,
                                    stream);
    }
  } else if (!w.o8.empty()) {
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
  Tensor ffn, up_out;
  if (nv_up) {
    // The [m, 2F] output in the op-by-op layout's place; with a side branch the bf16 norm output stays in y for t and
    // the codes go past the whole [L, 2F] output (the op-by-op ffn's place: free until situ_glu below), never inside
    // it: other tiles' epilogues write it while the GEMM still reads its input.
    up_out = rows_of(a.big, 0, l, 2 * kF, run.t0, run.t1);
    if (s_up != nullptr) {
      kernels::rms_norm_modulate_moss(x, w.norm2, vec(a.mod, 4), vec(a.mod, 3), y, stream);
      const kn::Nvfp4Delta d = nv_delta(*s_up, y, true);
      const kn::Nvfp4Act act = nv_act(static_cast<std::int8_t*>(a.big.data) + l * 2 * kF * 2 + run.t0 * (kD / 2));
      kn::quantize_nvfp4(bf(y), kD, act, m, static_cast<int>(kD), stream);
      kn::nvfp4_gemm(operands_nv(act, w.up_gate_nvfp4, m), nullptr, bf_mut(up_out), 2 * kF, stream, kn::Nvfp4Tile::kAuto, d);
    } else {
      const kn::Nvfp4Act act = nv_act(xq_y);
      kn::rms_norm_modulate_quantize_nvfp4(bf(x), kD, bf(w.norm2), bf(vec(a.mod, 4)), bf(vec(a.mod, 3)), act, m,
                                           static_cast<int>(kD), stream);
      kn::nvfp4_gemm(operands_nv(act, w.up_gate_nvfp4, m), nullptr, bf_mut(up_out), 2 * kF, stream);
    }
    if (!nv_ffn_fused) {
      ffn = rows_of(a.big, l * 2 * kF, l, kF, run.t0, run.t1);
      kernels::situ_glu(up_out, ffn, stream);
    }
  } else if (!w.up_gate4.empty() && s_up != nullptr) {
    // As the INT8 LoRA case: the bf16 norm output stays in y, the quantized input goes to the attention output's columns.
    ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
    kernels::rms_norm_modulate_moss(x, w.norm2, vec(a.mod, 4), vec(a.mod, 3), y, stream);
    kq4::convrot_quantize_int4(bf(y), kD, u8(xq_qkv + m * kD), kD / 2, scale, m, static_cast<int>(kD), stream);
    const kq::Int8Operands op = unpacked(xq_qkv + m * kD, xq_qkv, w.up_gate4);
    kl::int8_gemm_situ_glu_lora(op, prepare(*s_up, y), bf_mut(ffn), kF, stream);
  } else if (!w.up_gate4.empty() && via) {
    // The attention output's columns are free: packed input after its INT8 copy, as with a side branch.
    ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
    kq4::rms_norm_modulate_convrot_quantize_int4(bf(x), kD, bf(w.norm2), bf(vec(a.mod, 4)), bf(vec(a.mod, 3)),
                                                 u8(xq_qkv + m * kD), kD / 2, scale, m, static_cast<int>(kD), stream);
    kq::int8_gemm_situ_glu(unpacked(xq_qkv + m * kD, xq_qkv, w.up_gate4), bf_mut(ffn), kF, stream);
  } else if (!w.up_gate4.empty()) {
    ffn = rows_of(a.big, 0, l, kF, run.t0, run.t1);
    kq4::rms_norm_modulate_convrot_quantize_int4(bf(x), kD, bf(w.norm2), bf(vec(a.mod, 4)), bf(vec(a.mod, 3)), u8(xq_y),
                                                 kD / 2, scale, m, static_cast<int>(kD), stream);
    kq4::int4_gemm_situ_glu(operands4(u8(xq_y), kD / 2, scale, w.up_gate4, m), bf_mut(ffn), kF, stream);
  } else if (!w.up_gate8.empty() && s_up != nullptr) {
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
  // [L, kF] int8 (or packed int4) where the FFN leaves `big` free: after the fused [L, kF] ffn, or over the op-by-op
  // up_gate output. With an INT4 linear's LoRA the packed input goes to y instead (free here), the INT8 copy to `big`.
  std::int8_t* const base = up_gate_op_by_op || nv_up ? static_cast<std::int8_t*>(a.big.data)
                                                     : static_cast<std::int8_t*>(a.big.data) + l * kF * 2;
  std::int8_t* const xq = base + run.t0 * kF;
  if (!w.down_nvfp4.empty()) {
    // The codes in y (free here), as the INT8 / INT4 o-proj input.
    kn::Nvfp4Delta d;
    const kn::Nvfp4Act act = nv_act(xq_y);
    if (nv_ffn_fused) {
      kn::situ_glu_quantize_nvfp4(bf(up_out), 2 * kF, act, m, static_cast<int>(kF), stream);
    } else {
      if (s_down != nullptr) d = nv_delta(*s_down, ffn, false);
      kn::quantize_nvfp4(bf(ffn), ffn.stride[0], act, m, static_cast<int>(kF), stream);
    }
    kn::nvfp4_gemm_gated_residual(operands_nv(act, w.down_nvfp4, m), nullptr, bf(vec(a.mod, 5)), bf_mut(x), kD, stream,
                                  kn::Nvfp4Tile::kAuto, d);
  } else if (!w.down4.empty()) {
    kl::LoraSlices ld;
    if (s_down != nullptr) ld = prepare(*s_down, ffn);
    std::int8_t* const xq4 = u_down ? static_cast<std::int8_t*>(a.y.data) + run.t0 * (kF / 2) : xq;
    kq4::convrot_quantize_int4(bf(ffn), kF, u8(xq4), kF / 2, scale, m, static_cast<int>(kF), stream);
    if (s_down != nullptr) {
      kl::int8_gemm_gated_residual_lora(unpacked(xq4, xq, w.down4), nullptr, bf(vec(a.mod, 5)), ld, bf_mut(x), kD, stream);
    } else if (via) {
      kq::int8_gemm_gated_residual(unpacked(xq4, xq, w.down4), nullptr, bf(vec(a.mod, 5)), bf_mut(x), kD, stream);
    } else {
      kq4::int4_gemm_gated_residual(operands4(u8(xq4), kF / 2, scale, w.down4, m), nullptr, bf(vec(a.mod, 5)), bf_mut(x),
                                    kD, stream);
    }
  } else if (!w.down8.empty()) {
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
