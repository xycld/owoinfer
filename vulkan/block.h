// The plain compute path of one MOSS DiT block, bf16 throughout (src/models/moss_dit/dit_block_fused.h):
//   y    = rms_norm_modulate(x, norm1, scale_a, shift_a)
//   qkv  = y * Wqkv^T
//   q,k  = rope(rms_norm(q|k))  per head, in place
//   attn = attention(q, k, v)
//   x   += bf16(bf16(attn * Wo^T) * gate_a)
//   y    = rms_norm_modulate(x, norm2, scale_f, shift_f)
//   ffn  = situ(bf16(y * Wup^T), bf16(y * Wgate^T))
//   x   += bf16(bf16(ffn * Wdown^T) * gate_f)
// Two paths. kPlain runs on any device the runtime accepts, needs no cooperative matrices, and does not depend on the
// subgroup size; its operands need row strides that are multiples of 4 elements. kCoop runs the GEMMs on bf16 cooperative
// matrices (Capabilities::coop_bf16) and needs strides, N and K that are multiples of 8. Each record_* adds its own barrier.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "runtime.h"

namespace mi::vk {

inline constexpr int kDiTDim = 3072;
inline constexpr int kDiTHeads = 24;
inline constexpr int kDiTHeadDim = 128;
inline constexpr int kDiTFfnDim = 8192;

struct AttentionArgs {
  VkDeviceAddress q, k, v, o;
  std::uint32_t ldq, ldk, ldv, ldo;
  std::uint32_t hsq, hsk, hsv, hso;
  std::vector<std::int32_t> cu_q, cu_k;  // batch + 1 row offsets, on the host: sequences are dispatched one by one
  std::uint32_t heads;
  float scale;  // SDPA's 1 / sqrt(head_dim) for the DiT
};

enum class BlockPath { kCoop, kPlain };
const char* block_path_name(BlockPath path);

class BlockKernels {
 public:
  // kCoop where the device has it, unless `path` names one. Throws if the device cannot run the path asked for.
  explicit BlockKernels(Device& device);
  BlockKernels(Device& device, BlockPath path);
  ~BlockKernels();

  BlockPath path() const { return path_; }

  void record_norm_modulate(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress weight,
                            VkDeviceAddress scale, VkDeviceAddress shift, VkDeviceAddress out, std::uint32_t ldo,
                            std::uint32_t rows, std::uint32_t d) const;
  void record_gemm(VkCommandBuffer cb, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress x, std::uint32_t ldx,
                   VkDeviceAddress w, std::uint32_t ldw, std::uint32_t m, std::uint32_t n, std::uint32_t k) const;
  void record_gemm_gated_residual(VkCommandBuffer cb, VkDeviceAddress residual, std::uint32_t ldr, VkDeviceAddress x,
                                  std::uint32_t ldx, VkDeviceAddress w, std::uint32_t ldw, VkDeviceAddress gate,
                                  std::uint32_t m, std::uint32_t n, std::uint32_t k) const;
  void record_gemm_situ_glu(VkCommandBuffer cb, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress x,
                            std::uint32_t ldx, VkDeviceAddress w_up_gate, std::uint32_t ldw, std::uint32_t m,
                            std::uint32_t f, std::uint32_t k) const;
  // out = bf16(x W^T + bias), and residual += bf16(x W^T + bias): nn.Linear with a bias, alone or before a residual add.
  void record_gemm_bias(VkCommandBuffer cb, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress x, std::uint32_t ldx,
                        VkDeviceAddress w, std::uint32_t ldw, VkDeviceAddress bias, std::uint32_t m, std::uint32_t n,
                        std::uint32_t k) const;
  void record_gemm_bias_residual(VkCommandBuffer cb, VkDeviceAddress residual, std::uint32_t ldr, VkDeviceAddress x,
                                 std::uint32_t ldx, VkDeviceAddress w, std::uint32_t ldw, VkDeviceAddress bias,
                                 std::uint32_t m, std::uint32_t n, std::uint32_t k) const;
  // out = bf16(bf16(rms_norm(x)) * weight), without modulation.
  void record_norm(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t ldx, VkDeviceAddress weight, VkDeviceAddress out,
                   std::uint32_t ldo, std::uint32_t rows, std::uint32_t d) const;
  // Contiguous bf16 vectors of n elements, n a multiple of 4: out = silu(a); out = situ(hidden, gate).
  void record_silu(VkCommandBuffer cb, VkDeviceAddress a, VkDeviceAddress out, std::uint32_t n) const;
  void record_situ(VkCommandBuffer cb, VkDeviceAddress hidden, VkDeviceAddress gate, VkDeviceAddress out,
                   std::uint32_t n) const;
  void record_qk_norm_rope(VkCommandBuffer cb, VkDeviceAddress qkv, std::uint32_t ld, std::uint32_t tokens,
                           std::uint32_t heads, VkDeviceAddress q_norm, VkDeviceAddress k_norm, VkDeviceAddress table,
                           std::uint32_t table_ld) const;
  void record_attention(VkCommandBuffer cb, const AttentionArgs& args) const;

 private:
  BlockPath path_;
  std::unique_ptr<Pipeline> gemm_bf16_, gemm_gated_, gemm_situ_, gemm_bias_, gemm_bias_residual_, norm_modulate_,
      qk_norm_rope_, attention_, silu_, situ_;
};

struct BlockWeights {
  VkDeviceAddress norm1, norm2, q_norm, k_norm, qkv, o, up_gate, down;
};

// One run of `tokens` tokens: x, y [tokens, kDiTDim]; qkv [tokens, 3 kDiTDim]; attn [tokens, kDiTDim];
// ffn [tokens, kDiTFfnDim]; rope [tokens, 128]; mod [6 kDiTDim]: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f.
struct BlockBuffers {
  VkDeviceAddress x, y, qkv, attn, ffn, rope, mod;
  std::vector<std::int32_t> cu_seqlens;
  std::uint32_t tokens;
};

inline constexpr int kBlockOps = 8;
const char* block_op_name(int op);
void record_block_op(const BlockKernels& kernels, VkCommandBuffer cb, int op, const BlockWeights& w, const BlockBuffers& b);

}  // namespace mi::vk
