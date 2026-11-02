// The Qwen3.5 text encoder's kernels other than its GEMMs (those are BlockKernels'), in the engine's op order and rounding
// points (src/models/qwen35_text, src/kernels/gdn, src/kernels/basic). Plain compute: no cooperative matrices, 16-bit
// arithmetic or particular subgroup size, so both paths use them. Arguments are device addresses; strides are in elements.
// Each record_* adds its own barrier.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "runtime.h"

namespace mi::vk {

inline constexpr int kGdnKeyHeads = 16;
inline constexpr int kGdnValueHeads = 32;
inline constexpr int kGdnHeadDim = 128;
inline constexpr int kGdnConvDim = 2 * kGdnKeyHeads * kGdnHeadDim + kGdnValueHeads * kGdnHeadDim;  // [q | k | v], 8192
inline constexpr int kGdnChunk = 64;

// torch's threads per row for a last-dim fp32 reduction of `rows` rows of length d (setReduceConfig, 4-wide vector loads,
// 512 threads at most). Throws where torch would split a row across warps, which the norm does not reproduce.
std::uint32_t torch_reduce_threads(std::uint64_t rows, std::uint32_t d);

// Rows are (token, head) pairs: row (t, h) of x starts at x + t * ldx + h * hsx, likewise out and gate. out may alias x.
struct RmsNormArgs {
  VkDeviceAddress x = 0, weight = 0, gate = 0, out = 0;
  std::uint32_t ldx = 0, hsx = 0, ldo = 0, hso = 0, ldg = 0, hsg = 0;
  std::uint32_t tokens = 0, heads = 1, d = 0;
  bool gated = false;        // Qwen3_5RMSNormGated with `gate`; otherwise Qwen3_5RMSNorm
  VkDeviceAddress rope = 0;  // Qwen3_5RMSNorm only: [tokens, 64] bf16 table [cos(32) | sin(32)] applied after the norm
  std::uint32_t rope_ld = 64;
  float eps = 1e-6f;
};

// The gated delta rule's fp32 scratch for `tokens` packed tokens.
struct GdnWorkspace {
  VkDeviceAddress qn, kn, gcum, eg, el, a, attn, u, w;
  static std::uint64_t bytes(std::uint32_t tokens);
  static GdnWorkspace at(VkDeviceAddress base, std::uint32_t tokens);
};

// Sequences packed along the rows: sequence i is rows cu[i] .. cu[i + 1].
struct GdnArgs {
  VkDeviceAddress qkv = 0;  // conv output, bf16 rows [q | k | v] of stride ldqkv
  std::uint32_t ldqkv = 0;
  VkDeviceAddress g = 0;     // f32 [tokens, 32]
  VkDeviceAddress beta = 0;  // bf16 [tokens, 32]
  VkDeviceAddress out = 0;   // bf16 [tokens, 32, 128]
  GdnWorkspace ws{};
  std::vector<std::int32_t> cu;
};

// One sequence: q, k, v and o point at its first row. Query head h reads key/value head h / group. Rows q_first ..
// q_first + rows - 1 of the sequence are computed, each over keys 0 .. its own index.
struct TeAttentionArgs {
  VkDeviceAddress q = 0, k = 0, v = 0, o = 0;
  std::uint32_t ldq = 0, ldk = 0, ldv = 0, ldo = 0;
  std::uint32_t hsq = 0, hsk = 0, hsv = 0, hso = 0;
  std::uint32_t q_first = 0, rows = 0;
  std::uint32_t heads = 0, group = 1;
  float scale = 0.0625f;
};

class TeKernels {
 public:
  explicit TeKernels(Device& device);
  ~TeKernels();

  void record_rms_norm(VkCommandBuffer cb, const RmsNormArgs& args) const;
  // One sequence of `tokens` rows: y = silu(causal_conv1d(x)), x and y bf16 [tokens, 8192] with strides ldx and ldy.
  void record_conv_silu(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress weight,
                        VkDeviceAddress y, std::uint32_t ldy, std::uint32_t tokens) const;
  // ba: bf16 rows [b(32) | a(32)] of stride ldba; g f32 [tokens, 32]; beta bf16 [tokens, 32].
  void record_gates(VkCommandBuffer cb, VkDeviceAddress ba, std::uint32_t ldba, VkDeviceAddress a_log,
                    VkDeviceAddress dt_bias, VkDeviceAddress g, VkDeviceAddress beta, std::uint32_t tokens) const;
  void record_gated_delta_rule(VkCommandBuffer cb, const GdnArgs& args) const;
  void record_attention(VkCommandBuffer cb, const TeAttentionArgs& args) const;
  // out = bf16(bf16(silu(a)) * b) (silu) or bf16(a * bf16(sigmoid(b))); column c of b is
  // (c / block) * block_stride + c % block.
  enum class GateMul { kSilu, kSigmoid };
  void record_gate_mul(VkCommandBuffer cb, GateMul op, VkDeviceAddress a, std::uint32_t lda, VkDeviceAddress b,
                       std::uint32_t ldb, VkDeviceAddress out, std::uint32_t ldo, std::uint32_t rows, std::uint32_t cols,
                       std::uint32_t block, std::uint32_t block_stride) const;

 private:
  std::unique_ptr<Pipeline> norm_, norm_gated_, conv_, gates_, gdn_qk_, gdn_chunk_, gdn_solve_, gdn_state_, attention_,
      silu_mul_, sigmoid_mul_;
};

}  // namespace mi::vk
