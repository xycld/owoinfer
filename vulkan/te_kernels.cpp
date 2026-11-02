#include "te_kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mi::vk {
namespace {

const std::uint32_t kNormSpirv[] = {
#include "te_rms_norm.spv.inc"
};
const std::uint32_t kConvSpirv[] = {
#include "te_conv_silu.spv.inc"
};
const std::uint32_t kGatesSpirv[] = {
#include "te_gates.spv.inc"
};
const std::uint32_t kGdnQkSpirv[] = {
#include "te_gdn_qk.spv.inc"
};
const std::uint32_t kGdnChunkSpirv[] = {
#include "te_gdn_chunk.spv.inc"
};
const std::uint32_t kGdnSolveSpirv[] = {
#include "te_gdn_solve.spv.inc"
};
const std::uint32_t kGdnStateSpirv[] = {
#include "te_gdn_state.spv.inc"
};
const std::uint32_t kAttentionSpirv[] = {
#include "te_attention.spv.inc"
};
const std::uint32_t kGateMulSpirv[] = {
#include "te_gate_mul.spv.inc"
};

struct NormPush {
  std::uint64_t x, weight, gate, out, table;
  std::uint32_t ldx, hsx, ldo, hso, ldg, hsg, table_ld, d, bw;
  float factor, eps;
};
struct ConvPush {
  std::uint64_t x, w, y;
  std::uint32_t ldx, ldy, channels;
};
struct GatesPush {
  std::uint64_t ba, a_log, dt_bias, g, beta;
  std::uint32_t ldba, tokens;
};
struct GdnQkPush {
  std::uint64_t qkv, qn, kn;
  std::uint32_t ldqkv, tokens;
};
struct GdnChunkPush {
  std::uint64_t qn, kn, g, beta, gcum, eg, el, a, attn;
  std::uint32_t tokens, row0, len;
};
struct GdnSolvePush {
  std::uint64_t qkv, kn, beta, eg, a, u, w;
  std::uint32_t ldqkv, tokens, row0, len;
};
struct GdnStatePush {
  std::uint64_t qn, kn, gcum, eg, el, attn, u, w, out;
  std::uint32_t tokens, row0, len;
};
struct AttentionPush {
  std::uint64_t q, k, v, o;
  std::uint32_t ldq, ldk, ldv, ldo, hsq, hsk, hsv, hso, q_first, group;
  float log2_scale;
};
struct GateMulPush {
  std::uint64_t a, b, out;
  std::uint32_t lda, ldb, ldo, cols, block, block_stride;
};

constexpr std::uint32_t kMaxGroups = 65535;  // the guaranteed maxComputeWorkGroupCount per dimension

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("vulkan text encoder: " + message);
}

template <std::size_t N>
std::unique_ptr<Pipeline> pipeline(Device& device, const std::uint32_t (&spirv)[N], std::uint32_t push, int mode = -1) {
  std::vector<std::uint32_t> spec;
  if (mode >= 0) spec.push_back(static_cast<std::uint32_t>(mode));
  return std::make_unique<Pipeline>(device, spirv, sizeof(spirv), push, 0, spec);
}

std::uint32_t pow2_floor(std::uint64_t n) {
  std::uint32_t p = 1;
  while (static_cast<std::uint64_t>(p) * 2 <= n && p < (1u << 30)) p *= 2;
  return p;
}

}  // namespace

std::uint32_t torch_reduce_threads(std::uint64_t rows, std::uint32_t d) {
  require(d % 4 == 0 && d >= 128, "norm rows must be a multiple of 4 and at least 128 long");
  constexpr std::uint32_t kMaxThreads = 512;
  const std::uint32_t dim0 = d / 4;
  const std::uint32_t dim0_pow2 = dim0 < kMaxThreads ? pow2_floor(dim0) : kMaxThreads;
  const std::uint32_t dim1_pow2 = rows < kMaxThreads ? pow2_floor(rows) : kMaxThreads;
  std::uint32_t bw = std::min(dim0_pow2, 32u);
  const std::uint32_t bh = std::min(dim1_pow2, kMaxThreads / bw);
  bw = std::min(dim0_pow2, kMaxThreads / bh);
  const std::uint32_t per_thread = (d + bw - 1) / bw;
  require(per_thread < std::min(bh * 16, 256u), "row length " + std::to_string(d) + " needs torch's cross-warp split");
  return bw;
}

std::uint64_t GdnWorkspace::bytes(std::uint32_t tokens) {
  GdnWorkspace ws = at(0, tokens);
  return ws.w + static_cast<std::uint64_t>(kGdnValueHeads) * tokens * kGdnHeadDim * 4;
}

GdnWorkspace GdnWorkspace::at(VkDeviceAddress base, std::uint32_t tokens) {
  const auto align = [](std::uint64_t v) { return (v + 255) & ~std::uint64_t{255}; };
  const std::uint64_t t = tokens;
  const std::uint64_t key = align(kGdnKeyHeads * t * kGdnHeadDim * 4), per_head = align(kGdnValueHeads * t * 4),
                      chunk = align(kGdnValueHeads * t * kGdnChunk * 4), value = align(kGdnValueHeads * t * kGdnHeadDim * 4);
  GdnWorkspace ws{};
  ws.qn = base;
  ws.kn = ws.qn + key;
  ws.gcum = ws.kn + key;
  ws.eg = ws.gcum + per_head;
  ws.el = ws.eg + per_head;
  ws.a = ws.el + per_head;
  ws.attn = ws.a + chunk;
  ws.u = ws.attn + chunk;
  ws.w = ws.u + value;
  return ws;
}

TeKernels::TeKernels(Device& device) {
  norm_ = pipeline(device, kNormSpirv, sizeof(NormPush), 0);
  norm_gated_ = pipeline(device, kNormSpirv, sizeof(NormPush), 1);
  conv_ = pipeline(device, kConvSpirv, sizeof(ConvPush));
  gates_ = pipeline(device, kGatesSpirv, sizeof(GatesPush));
  gdn_qk_ = pipeline(device, kGdnQkSpirv, sizeof(GdnQkPush));
  gdn_chunk_ = pipeline(device, kGdnChunkSpirv, sizeof(GdnChunkPush));
  gdn_solve_ = pipeline(device, kGdnSolveSpirv, sizeof(GdnSolvePush));
  gdn_state_ = pipeline(device, kGdnStateSpirv, sizeof(GdnStatePush));
  attention_ = pipeline(device, kAttentionSpirv, sizeof(AttentionPush));
  silu_mul_ = pipeline(device, kGateMulSpirv, sizeof(GateMulPush), 0);
  sigmoid_mul_ = pipeline(device, kGateMulSpirv, sizeof(GateMulPush), 1);
}

TeKernels::~TeKernels() = default;

void TeKernels::record_rms_norm(VkCommandBuffer cb, const RmsNormArgs& a) const {
  require(a.d % 4 == 0 && a.ldx % 4 == 0 && a.hsx % 4 == 0 && a.ldo % 4 == 0 && a.hso % 4 == 0 &&
              (!a.gated || (a.ldg % 4 == 0 && a.hsg % 4 == 0)),
          "norm: d and strides must be multiples of 4");
  require(a.rope == 0 || (!a.gated && a.d >= 64 && a.rope_ld % 4 == 0), "norm: RoPE needs the plain norm and d >= 64");
  require(a.heads >= 1 && a.heads <= kMaxGroups && a.tokens <= kMaxGroups, "norm: too many rows");
  if (a.tokens == 0) return;
  const std::uint64_t rows = static_cast<std::uint64_t>(a.tokens) * a.heads;
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * a.d);
  const NormPush push{a.x,   a.weight, a.gated ? a.gate : 0, a.out,  a.gated ? 0 : a.rope, a.ldx,
                      a.hsx, a.ldo,    a.hso,                a.ldg,  a.hsg,                a.rope_ld,
                      a.d,   torch_reduce_threads(rows, a.d), factor, a.eps};
  (a.gated ? norm_gated_ : norm_)->dispatch(cb, &push, a.heads, a.tokens);
  compute_barrier(cb);
}

void TeKernels::record_conv_silu(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress weight,
                                 VkDeviceAddress y, std::uint32_t ldy, std::uint32_t tokens) const {
  require(ldx % 4 == 0 && ldy % 4 == 0 && tokens <= kMaxGroups, "conv: strides must be multiples of 4");
  if (tokens == 0) return;
  const ConvPush push{x, weight, y, ldx, ldy, kGdnConvDim};
  conv_->dispatch(cb, &push, kGdnConvDim / 4 / 64, tokens);
  compute_barrier(cb);
}

void TeKernels::record_gates(VkCommandBuffer cb, VkDeviceAddress ba, std::uint32_t ldba, VkDeviceAddress a_log,
                             VkDeviceAddress dt_bias, VkDeviceAddress g, VkDeviceAddress beta, std::uint32_t tokens) const {
  require(ldba % 2 == 0, "gates: the stride must be even");
  if (tokens == 0) return;
  const GatesPush push{ba, a_log, dt_bias, g, beta, ldba, tokens};
  gates_->dispatch(cb, &push, (tokens * 16 + 63) / 64, 1);
  compute_barrier(cb);
}

void TeKernels::record_gated_delta_rule(VkCommandBuffer cb, const GdnArgs& a) const {
  require(a.cu.size() >= 2 && a.cu.front() == 0, "gated delta rule: cu must start at 0 and hold a sequence");
  const auto tokens = static_cast<std::uint32_t>(a.cu.back());
  require(tokens <= kMaxGroups && a.ldqkv % 4 == 0, "gated delta rule: too many tokens or a bad stride");
  if (tokens == 0) return;
  const GdnQkPush qk{a.qkv, a.ws.qn, a.ws.kn, a.ldqkv, tokens};
  gdn_qk_->dispatch(cb, &qk, kGdnKeyHeads, tokens);
  compute_barrier(cb);
  const auto for_each = [&](auto&& f) {
    for (std::size_t i = 0; i + 1 < a.cu.size(); ++i) {
      require(a.cu[i + 1] >= a.cu[i], "gated delta rule: cu must not decrease");
      const auto row0 = static_cast<std::uint32_t>(a.cu[i]), len = static_cast<std::uint32_t>(a.cu[i + 1] - a.cu[i]);
      if (len > 0) f(row0, len, (len + kGdnChunk - 1) / kGdnChunk);
    }
  };
  for_each([&](std::uint32_t row0, std::uint32_t len, std::uint32_t chunks) {
    const GdnChunkPush push{a.ws.qn, a.ws.kn, a.g, a.beta, a.ws.gcum, a.ws.eg, a.ws.el, a.ws.a, a.ws.attn, tokens, row0, len};
    gdn_chunk_->dispatch(cb, &push, chunks, kGdnValueHeads);
  });
  compute_barrier(cb);
  for_each([&](std::uint32_t row0, std::uint32_t len, std::uint32_t chunks) {
    const GdnSolvePush push{a.qkv, a.ws.kn, a.beta, a.ws.eg, a.ws.a, a.ws.u, a.ws.w, a.ldqkv, tokens, row0, len};
    gdn_solve_->dispatch(cb, &push, chunks * 4, kGdnValueHeads);
  });
  compute_barrier(cb);
  for_each([&](std::uint32_t row0, std::uint32_t len, std::uint32_t) {
    const GdnStatePush push{a.ws.qn, a.ws.kn, a.ws.gcum, a.ws.eg, a.ws.el, a.ws.attn, a.ws.u, a.ws.w, a.out, tokens, row0, len};
    gdn_state_->dispatch(cb, &push, kGdnHeadDim / 16, kGdnValueHeads);
  });
  compute_barrier(cb);
}

void TeKernels::record_attention(VkCommandBuffer cb, const TeAttentionArgs& a) const {
  require(a.ldq % 4 == 0 && a.ldk % 4 == 0 && a.ldv % 4 == 0 && a.ldo % 4 == 0 && a.hsq % 4 == 0 && a.hsk % 4 == 0 &&
              a.hsv % 4 == 0 && a.hso % 4 == 0,
          "attention: strides must be multiples of 4");
  require(a.group >= 1 && a.heads % a.group == 0 && a.rows <= kMaxGroups, "attention: bad head grouping or too many rows");
  if (a.rows == 0) return;
  // In double then rounded, as FlashAttention (float * M_LOG2E).
  const float log2_scale = static_cast<float>(static_cast<double>(a.scale) * 1.4426950408889634074);
  const AttentionPush push{a.q, a.k, a.v, a.o, a.ldq, a.ldk, a.ldv, a.ldo, a.hsq, a.hsk, a.hsv, a.hso, a.q_first, a.group, log2_scale};
  attention_->dispatch(cb, &push, a.heads, a.rows);
  compute_barrier(cb);
}

void TeKernels::record_gate_mul(VkCommandBuffer cb, GateMul op, VkDeviceAddress a, std::uint32_t lda, VkDeviceAddress b,
                                std::uint32_t ldb, VkDeviceAddress out, std::uint32_t ldo, std::uint32_t rows,
                                std::uint32_t cols, std::uint32_t block, std::uint32_t block_stride) const {
  require(lda % 4 == 0 && ldb % 4 == 0 && ldo % 4 == 0 && cols % 4 == 0 && block % 4 == 0 && block_stride % 4 == 0 &&
              block > 0 && rows <= kMaxGroups,
          "gate_mul: strides, columns and blocks must be multiples of 4");
  if (rows == 0 || cols == 0) return;
  const GateMulPush push{a, b, out, lda, ldb, ldo, cols, block, block_stride};
  (op == GateMul::kSilu ? silu_mul_ : sigmoid_mul_)->dispatch(cb, &push, (cols / 4 + 63) / 64, rows);
  compute_barrier(cb);
}

}  // namespace mi::vk
