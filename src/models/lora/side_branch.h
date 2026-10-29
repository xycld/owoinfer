// Device side branch of the runtime LoRA stack: the low-rank adapters of the DiT block GEMMs (LoraStack adapters
// with side_branch set), rank-concatenated so that one down GEMM and one up GEMM per output slice serve every LoRA.
//
// Per block and site (the GEMM the block runs: fused qkv, o, ffn_up_gate, ffn_down), for its input x [m, k]:
//   t32 = x · Aᵀ                       cuBLASLt, bf16 in, fp32 accumulate and out; A = [A_1; A_2; ...] [R, k]
//   t   = bf16(t32 * coef)             per rank column j: coef_j = f32(strength * alpha / rank) of its adapter
// (compute_t), then the LoRA GEMMs of kernels/lora/lora_gemm.h add t · B_sᵀ per output slice (B_s = the slice's
// adapters' B side by side) as a second mainloop into the base GEMM's fp32 accumulators, before its epilogue op
// (lora_slices() gives their operand; Δ is never stored):
//   qkv:          qkv = Wqkv·x + Δ                   (then QK-norm + RoPE: Δ is added before the norm)
//   o, ffn_down:  x  += gate ⊙ (W·in + Δ)            (gated residual)
//   ffn_up_gate:  ffn = SiTU-GLU(Wup·y + Δ)          (B's rows are stored in the SiTU GEMM's permuted column order:
//                                                     row c holds weight row up_gate_gemm_row(c, F))
// The qkv site has q, k, v slices at output columns 0, dim, 2 dim. Adapters written on a fused qkv (one A, B of
// 3 dim rows) keep one shared range of t columns that all three slices read (their first range); adapters on q, k or
// v alone form each slice's own second range. Each range is padded to a multiple of 8 columns (zero rows of A, zero
// coef, zero columns of B), so every operand stays 16-byte aligned.
//
// Strength changes are instant: update_strengths() rewrites the per-column coefficients (one batched upload, no other
// data moves). A site whose coefficients are all zero is inactive: site() returns null and the block runs its plain
// kernels, bit-identical to no LoRA. A strength-0 LoRA inside an active site contributes exact zeros.
//
// Memory: everything lives in one device allocation (device_bytes()), resident for every block whether the base
// block is resident or streamed: rank-64 adapters on all 6 GEMMs are 7.1 MB per block (3.1 % of a bf16 block, 6.3 %
// of an INT8 one), so streaming them would add PCIe traffic and synchronization for no measurable VRAM gain.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "kernels/lora/lora_gemm.h"
#include "models/lora/lora.h"

namespace mi::lora {

enum class Site : std::uint8_t { kQkv, kO, kUpGate, kDown };
inline constexpr int kSites = 4;
const char* site_name(Site s);  // "qkv", "o", "ffn_up_gate", "ffn_down"

struct SiteBranch {
  struct Slice {
    std::int64_t col0 = 0, cols = 0;    // t columns of the first range (multiples of 8)
    std::int64_t col0b = 0, colsb = 0;  // second range (0 columns: none)
    std::int64_t row0 = 0, rows = 0;    // output columns this slice adds to
    Tensor b;                           // bf16 [rows, cols + colsb], contiguous
  };
  std::int64_t n = 0, k = 0;  // the GEMM's output columns (qkv 3 dim, ffn_up_gate 2F) and input columns
  std::int64_t rank = 0;      // R: rows of a, columns of t
  Tensor a;                   // bf16 [R, k], contiguous
  Tensor coef;                // f32 [R]
  std::vector<Slice> slices;  // ascending row0, disjoint
};

class SideBranches {
 public:
  // Uploads the side-branch adapters of `stack` (factors rounded to bf16) with its current strengths. Synchronous.
  // `stack` must outlive this object if update_strengths() is used.
  explicit SideBranches(const LoraStack& stack);
  SideBranches(const SideBranches&) = delete;
  SideBranches& operator=(const SideBranches&) = delete;

  // The layout `stack` would get, without allocating anything (a plan-only pipeline: mi_check); every factor the upload
  // reads is read too (and dropped), so a file that fails to read fails here as well:
  // site() (null or not; a branch it returns has no tensors), any_active(), max_rank() and device_bytes() answer as for
  // the uploaded object, with the same refusal of a non-finite coefficient; update_strengths() is refused.
  struct LayoutOnly {};
  SideBranches(const LoraStack& stack, LayoutOnly);

  // Device bytes the constructor allocates for `stack` (for the planner).
  static std::size_t device_bytes(const LoraStack& stack);
  std::size_t device_bytes() const { return buffer_.size() + layout_bytes_; }

  // The site's branch, or null when no adapter with a nonzero coefficient targets it.
  const SiteBranch* site(int block, Site s) const;
  bool any_active(int block) const;
  bool any_active() const;
  int blocks() const { return blocks_; }
  // Largest R over the sites (size of t and of t_workspace_bytes).
  std::int64_t max_rank() const { return max_rank_; }

  // Re-reads the strengths of `stack`, which must be the object this was built from (std::invalid_argument
  // otherwise, as for a non-finite coefficient; nothing is uploaded then). Every coefficient is computed first, then
  // all are uploaded in one synchronous copy: call it between forwards, never while a forward that reads them is in
  // flight or a graph is being captured.
  void update_strengths(const LoraStack& stack);
  // A process-wide unique value, renewed whenever the set of active sites changes (and at construction): CUDA graphs
  // of blocks key on it, since the kernel sequence differs. Two SideBranches objects never share a value.
  std::uint64_t generation() const { return generation_; }

 private:
  struct Column {  // a run of rank columns of one adapter (host bookkeeping for strength updates)
    std::size_t adapter;
    std::int64_t col0, rank;
  };
  struct Entry {
    SiteBranch branch;
    std::vector<Column> columns;
    bool present = false, active = false;
  };
  void upload_coefs(const LoraStack& stack);

  const LoraStack* stack_ = nullptr;
  std::size_t adapters_ = 0;
  int blocks_ = 0;
  std::int64_t max_rank_ = 0;
  std::vector<Entry> entries_;  // [block * kSites + site]
  std::size_t coef_bytes_ = 0;  // every site's coefficients, contiguous at the start of buffer_
  DeviceBuffer buffer_;
  std::size_t layout_bytes_ = 0;  // LayoutOnly: the bytes the upload would allocate (buffer_ stays empty)
  std::uint64_t generation_ = 0;
};

// t (bf16 [m, site.rank] rows with one row stride, 16-byte aligned, stride a multiple of 8) for the site's input x
// (bf16 [m, site.k] rows). workspace: t_workspace_bytes(m, site.rank) bytes, 256-byte aligned (t32). gemm / gemm_ws:
// the model's cuBLASLt handle and workspace. Launches only on `stream`; graph-capturable.
std::size_t t_workspace_bytes(std::int64_t m, std::int64_t rank);
void compute_t(const SiteBranch& site, const Tensor& x, const Tensor& t, void* workspace, std::size_t workspace_bytes,
               kernels::Gemm& gemm, void* gemm_ws, std::size_t gemm_ws_bytes, cudaStream_t stream);

// The LoRA operand of the fused GEMMs for this site and its t (as written by compute_t).
kernels::lora::LoraSlices lora_slices(const SiteBranch& site, const Tensor& t);

// The LoRA term of the NVFP4 GEMMs (kernels/nvfp4 E1L / E2L, which add a stored Δ in their epilogue): Δ = t · B_sᵀ for
// every slice of the site, one cuBLASLt GEMM per slice (bf16 in, fp32 accumulation, rounded to bf16 once), into delta
// (CUDA bf16 [m, site.n] rows, 16-byte aligned, row stride a multiple of 8); the columns no slice covers are zeroed.
// A slice with two t ranges gathers them into scratch first. natural_order (the ffn_up_gate site): Δ's columns in the
// stored [hidden | gate] order of the weight instead of the SiTU GEMMs' permuted order B is kept in (B's rows are
// reordered in scratch: row up_gate_gemm_row(c, F) <- row c). scratch: delta_scratch_bytes(m, site.rank, site.n),
// 256-byte aligned. Launches only on `stream` (copies, memsets and GEMMs; no allocation, no synchronization).
std::size_t delta_scratch_bytes(std::int64_t m, std::int64_t rank, std::int64_t n);
void compute_delta(const SiteBranch& site, const Tensor& t, const Tensor& delta, bool natural_order, void* scratch,
                   std::size_t scratch_bytes, kernels::Gemm& gemm, void* gemm_ws, std::size_t gemm_ws_bytes,
                   cudaStream_t stream);

}  // namespace mi::lora
