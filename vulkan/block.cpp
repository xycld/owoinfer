#include "block.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace mi::vk {
namespace {

const std::uint32_t kGemmSpirv[] = {
#include "gemm_plain.spv.inc"
};
const std::uint32_t kGemmCoopSpirv[] = {
#include "gemm_bf16_coop.spv.inc"
};
const std::uint32_t kNormSpirv[] = {
#include "norm_modulate.spv.inc"
};
const std::uint32_t kQkSpirv[] = {
#include "qk_norm_rope.spv.inc"
};
const std::uint32_t kAttentionSpirv[] = {
#include "attention_plain.spv.inc"
};
const std::uint32_t kAttentionCoopSpirv[] = {
#include "attention_coop.spv.inc"
};

struct GemmPush {
  std::uint64_t a, b, out, gate;
  std::uint32_t m, n, k, lda, ldb, ldo;
};
struct NormPush {
  std::uint64_t x, weight, scale, shift, out;
  std::uint32_t ldx, ldo, d;
  float factor;
};
struct QkPush {
  std::uint64_t qkv, q_norm, k_norm, table;
  std::uint32_t ld, table_ld, heads;
  float factor;
};
struct AttentionPush {
  std::uint64_t q, k, v, o;
  std::uint32_t ldq, ldk, ldv, ldo, hsq, hsk, hsv, hso, q_row0, k_row0, seqlen_k;
  float log2_scale;
};

struct CoopAttentionPush {
  std::uint64_t q, k, v, o;
  std::uint32_t ldq, ldk, ldv, ldo, hsq, hsk, hsv, hso, q_row0, k_row0, seqlen_q, seqlen_k;
  float log2_scale;
};
constexpr std::uint32_t kCoopAttentionRows = 64;

enum GemmMode : std::uint32_t { kF32 = 0, kBF16 = 1, kGated = 2, kSitu = 3 };

constexpr std::uint32_t kCoopSubgroup = 32;
constexpr std::uint32_t kCoopTileRows = 128, kCoopTileCols = 128;

std::unique_ptr<Pipeline> gemm_pipeline(Device& device, BlockPath path, GemmMode mode) {
  const auto m = static_cast<std::uint32_t>(mode);
  if (path == BlockPath::kCoop) {
    return std::make_unique<Pipeline>(device, kGemmCoopSpirv, sizeof(kGemmCoopSpirv), sizeof(GemmPush), kCoopSubgroup,
                                      std::vector<std::uint32_t>{m});
  }
  return std::make_unique<Pipeline>(device, kGemmSpirv, sizeof(kGemmSpirv), sizeof(GemmPush), 0,
                                    std::vector<std::uint32_t>{1u, m});
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("vulkan block: " + message);
}

constexpr std::uint32_t kGemmTileRows = 8, kGemmTileCols = 32;

void dispatch_gemm(const Pipeline& pipeline, BlockPath path, GemmMode mode, VkCommandBuffer cb, const GemmPush& push) {
  const std::uint32_t align = path == BlockPath::kCoop ? 8 : 4;
  require(push.m > 0 && push.n > 0 && push.k > 0, "bad GEMM shape");
  require(push.n % align == 0 && push.k % align == 0 && push.lda % align == 0 && push.ldb % align == 0 &&
              push.ldo % align == 0,
          "GEMM N, K and strides must be multiples of " + std::to_string(align));
  if (path == BlockPath::kCoop) {
    // The shader numbers tiles y * x_count + x and walks them in bands of N tiles.
    const std::uint32_t cols = mode == kSitu ? kCoopTileCols / 2 : kCoopTileCols;
    pipeline.dispatch(cb, &push, (push.n + cols - 1) / cols, (push.m + kCoopTileRows - 1) / kCoopTileRows);
  } else {
    pipeline.dispatch(cb, &push, (push.n + kGemmTileCols - 1) / kGemmTileCols, (push.m + kGemmTileRows - 1) / kGemmTileRows);
  }
  compute_barrier(cb);
}

}  // namespace

const char* block_path_name(BlockPath path) { return path == BlockPath::kCoop ? "coop" : "plain"; }

BlockKernels::BlockKernels(Device& device)
    : BlockKernels(device, device.info().caps.coop_bf16 ? BlockPath::kCoop : BlockPath::kPlain) {}

BlockKernels::BlockKernels(Device& device, BlockPath path) : path_(path) {
  if (path == BlockPath::kCoop && !device.info().caps.coop_bf16) {
    throw std::runtime_error(device.info().name + " cannot run the cooperative-matrix DiT block: " +
                             device.info().caps.coop_bf16_reason);
  }
  gemm_bf16_ = gemm_pipeline(device, path, kBF16);
  gemm_gated_ = gemm_pipeline(device, path, kGated);
  gemm_situ_ = gemm_pipeline(device, path, kSitu);
  norm_modulate_ = std::make_unique<Pipeline>(device, kNormSpirv, sizeof(kNormSpirv), sizeof(NormPush));
  qk_norm_rope_ = std::make_unique<Pipeline>(device, kQkSpirv, sizeof(kQkSpirv), sizeof(QkPush));
  if (path == BlockPath::kCoop) {
    attention_ = std::make_unique<Pipeline>(device, kAttentionCoopSpirv, sizeof(kAttentionCoopSpirv),
                                            sizeof(CoopAttentionPush), kCoopSubgroup);
  } else {
    attention_ = std::make_unique<Pipeline>(device, kAttentionSpirv, sizeof(kAttentionSpirv), sizeof(AttentionPush));
  }
}

BlockKernels::~BlockKernels() = default;

void BlockKernels::record_norm_modulate(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress weight,
                                        VkDeviceAddress scale, VkDeviceAddress shift, VkDeviceAddress out,
                                        std::uint32_t ldo, std::uint32_t rows, std::uint32_t d) const {
  require(d % 4 == 0 && ldx % 4 == 0 && ldo % 4 == 0, "norm: d and strides must be multiples of 4");
  if (rows == 0) return;
  // torch's MeanOps factor.
  const float factor = static_cast<float>(rows) / static_cast<float>(static_cast<std::uint64_t>(rows) * d);
  const NormPush push{x, weight, scale, shift, out, ldx, ldo, d, factor};
  norm_modulate_->dispatch(cb, &push, rows, 1);
  compute_barrier(cb);
}

void BlockKernels::record_gemm(VkCommandBuffer cb, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress x,
                               std::uint32_t ldx, VkDeviceAddress w, std::uint32_t ldw, std::uint32_t m, std::uint32_t n,
                               std::uint32_t k) const {
  dispatch_gemm(*gemm_bf16_, path_, kBF16, cb, GemmPush{x, w, out, 0, m, n, k, ldx, ldw, ldo});
}

void BlockKernels::record_gemm_gated_residual(VkCommandBuffer cb, VkDeviceAddress residual, std::uint32_t ldr,
                                              VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress w, std::uint32_t ldw,
                                              VkDeviceAddress gate, std::uint32_t m, std::uint32_t n,
                                              std::uint32_t k) const {
  dispatch_gemm(*gemm_gated_, path_, kGated, cb, GemmPush{x, w, residual, gate, m, n, k, ldx, ldw, ldr});
}

void BlockKernels::record_gemm_situ_glu(VkCommandBuffer cb, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress x,
                                        std::uint32_t ldx, VkDeviceAddress w_up_gate, std::uint32_t ldw, std::uint32_t m,
                                        std::uint32_t f, std::uint32_t k) const {
  dispatch_gemm(*gemm_situ_, path_, kSitu, cb, GemmPush{x, w_up_gate, out, 0, m, f, k, ldx, ldw, ldo});
}

void BlockKernels::record_qk_norm_rope(VkCommandBuffer cb, VkDeviceAddress qkv, std::uint32_t ld, std::uint32_t tokens,
                                       std::uint32_t heads, VkDeviceAddress q_norm, VkDeviceAddress k_norm,
                                       VkDeviceAddress table, std::uint32_t table_ld) const {
  require(ld % 4 == 0 && table_ld % 4 == 0, "qk_norm_rope: strides must be multiples of 4");
  if (tokens == 0) return;
  // rms_norm_moss on a [L, heads, 128] head view sees L * heads rows.
  const std::uint64_t rows = static_cast<std::uint64_t>(tokens) * heads;
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * kDiTHeadDim);
  const QkPush push{qkv, q_norm, k_norm, table, ld, table_ld, heads, factor};
  qk_norm_rope_->dispatch(cb, &push, 2 * heads, tokens);
  compute_barrier(cb);
}

void BlockKernels::record_attention(VkCommandBuffer cb, const AttentionArgs& a) const {
  require(a.cu_q.size() == a.cu_k.size() && a.cu_q.size() >= 2, "attention: cu_q and cu_k must describe the same batch");
  // In double then rounded, as FlashAttention (float * M_LOG2E): keeps the exp2 argument bit-identical.
  const float log2_scale = static_cast<float>(static_cast<double>(a.scale) * 1.4426950408889634074);
  if (path_ == BlockPath::kCoop) {
    require(a.ldq % 8 == 0 && a.ldk % 8 == 0 && a.ldv % 8 == 0 && a.ldo % 8 == 0 && a.hsq % 8 == 0 && a.hsk % 8 == 0 &&
                a.hsv % 8 == 0 && a.hso % 8 == 0,
            "attention: strides must be multiples of 8 on the cooperative-matrix path");
  }
  for (std::size_t b = 0; b + 1 < a.cu_q.size(); ++b) {
    const std::int32_t seqlen_q = a.cu_q[b + 1] - a.cu_q[b], seqlen_k = a.cu_k[b + 1] - a.cu_k[b];
    if (seqlen_q <= 0) continue;
    require(seqlen_k > 0, "attention: a sequence with queries has no keys");
    if (path_ == BlockPath::kCoop) {
      const CoopAttentionPush push{a.q, a.k, a.v, a.o, a.ldq, a.ldk, a.ldv, a.ldo, a.hsq, a.hsk, a.hsv, a.hso,
                                   static_cast<std::uint32_t>(a.cu_q[b]), static_cast<std::uint32_t>(a.cu_k[b]),
                                   static_cast<std::uint32_t>(seqlen_q), static_cast<std::uint32_t>(seqlen_k), log2_scale};
      attention_->dispatch(cb, &push, a.heads,
                           (static_cast<std::uint32_t>(seqlen_q) + kCoopAttentionRows - 1) / kCoopAttentionRows);
      continue;
    }
    const AttentionPush push{a.q,   a.k,   a.v,   a.o,   a.ldq,
                             a.ldk, a.ldv, a.ldo, a.hsq, a.hsk,
                             a.hsv, a.hso, static_cast<std::uint32_t>(a.cu_q[b]), static_cast<std::uint32_t>(a.cu_k[b]),
                             static_cast<std::uint32_t>(seqlen_k), log2_scale};
    attention_->dispatch(cb, &push, a.heads, static_cast<std::uint32_t>(seqlen_q));
  }
  compute_barrier(cb);
}

const char* block_op_name(int op) {
  static constexpr const char* kNames[kBlockOps] = {"norm1+modulate",   "qkv gemm",       "qk_norm_rope",      "attention",
                                                    "o gemm+gated_res", "norm2+modulate", "up_gate gemm+situ", "down gemm+gated_res"};
  return kNames[op];
}

void record_block_op(const BlockKernels& k, VkCommandBuffer cb, int op, const BlockWeights& w, const BlockBuffers& b) {
  constexpr std::uint32_t ldx = kDiTDim, ldq = 3 * kDiTDim, ldf = kDiTFfnDim, dim = kDiTDim, head = kDiTHeadDim;
  const auto mod = [&](int i) { return b.mod + static_cast<VkDeviceAddress>(i) * kDiTDim * 2; };
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kDiTHeadDim)));
  switch (op) {
    case 0: k.record_norm_modulate(cb, b.x, ldx, w.norm1, mod(1), mod(0), b.y, ldx, b.tokens, dim); break;
    case 1: k.record_gemm(cb, b.qkv, ldq, b.y, ldx, w.qkv, dim, b.tokens, 3 * dim, dim); break;
    case 2: k.record_qk_norm_rope(cb, b.qkv, ldq, b.tokens, kDiTHeads, w.q_norm, w.k_norm, b.rope, head); break;
    case 3: {
      const VkDeviceAddress kv = b.qkv + static_cast<VkDeviceAddress>(dim) * 2;
      const AttentionArgs args{b.qkv, kv, kv + static_cast<VkDeviceAddress>(dim) * 2, b.attn, ldq, ldq, ldq, ldx,
                               head, head, head, head, b.cu_seqlens, b.cu_seqlens, kDiTHeads, scale};
      k.record_attention(cb, args);
      break;
    }
    case 4: k.record_gemm_gated_residual(cb, b.x, ldx, b.attn, ldx, w.o, dim, mod(2), b.tokens, dim, dim); break;
    case 5: k.record_norm_modulate(cb, b.x, ldx, w.norm2, mod(4), mod(3), b.y, ldx, b.tokens, dim); break;
    case 6: k.record_gemm_situ_glu(cb, b.ffn, ldf, b.y, ldx, w.up_gate, dim, b.tokens, kDiTFfnDim, dim); break;
    case 7: k.record_gemm_gated_residual(cb, b.x, ldx, b.ffn, ldf, w.down, kDiTFfnDim, mod(5), b.tokens, dim, kDiTFfnDim); break;
    default: throw std::invalid_argument("vulkan block: no such op");
  }
}

}  // namespace mi::vk
