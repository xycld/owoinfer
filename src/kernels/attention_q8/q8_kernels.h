// Internal launch interface of kernels/attention_q8 (attention_q8.cu builds these from the public arguments).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace mi::attention_q8::detail {

inline constexpr int kHeadDim = 128;
inline constexpr int kCtaQ = 128;      // query rows per attention CTA
inline constexpr int kCtaK = 64;       // keys per K/V tile (also the K quantization block and the V padding)
inline constexpr int kStatsChunk = 512;  // keys per partial K-sum / V-amax chunk

// Quantization and layout pre-pass (q8_preprocess.cu). Element strides; inputs bf16, outputs contiguous.
struct PrepParams {
  const void* q = nullptr;
  const void* k = nullptr;
  const void* v = nullptr;
  std::int64_t q_row = 0, q_head = 0, k_row = 0, k_head = 0, v_row = 0, v_head = 0;
  const int* cu_q = nullptr;
  const int* cu_k = nullptr;
  int batch = 0, heads = 0, kv_heads = 0, max_seqlen = 0;
  int warp_q = 32;  // Q quantization rows (the attention kernel's WARP_Q)
  bool fp8 = false;  // V: fp8 transposed/padded (true) or fp16 [total_k, kv_heads, 128] (false)
  bool v_amax = false;  // fp16 V: also measure the per-channel |V| max into v_scale (debug checks)
  float v_range = 0;    // fp8: e4m3 V range; fp16 with v_amax: 1 (v_scale = amax)
  std::int64_t padded_k = 0;  // fp8: columns per (head, channel) row of the V buffer
  int chunks = 0;             // stats chunks per sequence: ceil(max_seqlen / kStatsChunk)
  // Outputs (workspace).
  std::int8_t* q_i8 = nullptr;
  std::int8_t* k_i8 = nullptr;
  void* v_out = nullptr;
  float* q_scale = nullptr;
  float* k_scale = nullptr;
  float* k_mean = nullptr;
  float* v_scale = nullptr;
  float* v_recp = nullptr;  // fp8: v_range / amax (the quantization multiplier)
  float* part_sum = nullptr;   // [batch, kv_heads, chunks, 128]
  float* part_amax = nullptr;  // [batch, kv_heads, chunks, 128] (fp8)
};
void preprocess(const PrepParams& p, cudaStream_t stream);

// The attention kernels (q8_attn_f16.cu, q8_attn_f8.cu). Strides in elements of the respective buffer.
struct AttnParams {
  const std::int8_t* q = nullptr;
  const std::int8_t* k = nullptr;
  const void* v = nullptr;  // fp16 [total_k, kv_heads, 128] or e4m3 [kv_heads, 128, padded_k]
  void* o = nullptr;        // bf16
  const float* q_scale = nullptr;
  const float* k_scale = nullptr;
  const float* v_scale = nullptr;
  const int* cu_q = nullptr;
  const int* cu_k = nullptr;
  int batch = 0, heads = 0, kv_groups = 1, max_seqlen = 0;
  std::uint32_t k_scale_blocks = 0;
  std::uint32_t stride_seq_q = 0, stride_h_q = 0, stride_seq_k = 0, stride_h_k = 0;
  std::uint32_t stride_seq_v = 0, stride_h_v = 0;  // fp16 V: row, head; fp8 V: channel (stride_d), head
  std::uint32_t stride_seq_o = 0, stride_h_o = 0;
  float sm_scale = 0;
  float p_offset = 0;  // fp8: log2(p_range)
};
void run_int8_fp16(const AttnParams& p, bool f32_accum, cudaStream_t stream);
void run_int8_fp8(const AttnParams& p, bool f32_accum, cudaStream_t stream);

// The compute capability a kernel's image was compiled for, as loaded on the current device: min(ptxVersion,
// binaryVersion) of cudaFuncGetAttributes. ptxVersion is the virtual architecture (__CUDA_ARCH__ of the code), which a
// PTX image JIT-compiled for a newer device keeps; binaryVersion is the SASS target. 0 when no image loads.
template <class Kernel>
int kernel_arch(Kernel* kernel) {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, kernel) != cudaSuccess) {
    (void)cudaGetLastError();  // not sticky: clear it so a later launch check does not report it
    return 0;
  }
  return std::min(a.ptxVersion, a.binaryVersion);
}
int int8_fp16_kernel_arch(bool f32_accum);
int int8_fp8_kernel_arch(bool f32_accum);  // 0 for !f32_accum when !int8_fp8_f16_accum_compiled()
// False when nvcc < 12.8 compiled this library: it has no fp16-accumulating e4m3 MMA (the vendored wrapper is a
// __brkpt there), so the kInt8Fp8 kernel is not built and run_int8_fp8(p, false, ...) throws std::runtime_error.
bool int8_fp8_f16_accum_compiled();

}  // namespace mi::attention_q8::detail
