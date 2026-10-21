#include "kernels/attention/flash_attention.h"

#include <array>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention/flash_fwd_kernels.h"

namespace mi::attention {
namespace {

constexpr double kLog2e = 1.4426950408889634074;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("flash_fwd: " + what);
}

// 128-bit cp.async loads and stores need 16-byte aligned rows and heads.
void check_qkvo(const Tensor& t, const char* name) {
  require(t.device == Device::kCUDA && t.dtype == DType::kBF16 && t.rank == 3,
          std::string(name) + " must be a bf16 [tokens, heads, head_dim] CUDA tensor, got " + t.describe());
  require(t.stride[2] == 1, std::string(name) + " head_dim must be contiguous");
  require(t.stride[0] % 8 == 0 && t.stride[1] % 8 == 0 && reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0,
          std::string(name) + " needs 16-byte aligned base and row/head strides that are multiples of 8");
}

void check_cu_seqlens(const Tensor& t, const char* name) {
  require(t.device == Device::kCUDA && t.dtype == DType::kI32 && t.rank == 1 && t.dim(0) >= 2 && t.stride[0] == 1,
          std::string(name) + " must be a contiguous int32 [batch + 1] CUDA tensor, got " + t.describe());
}

// Q tile + double-buffered K and V tiles in bf16 (Share_Q_K_smem is off for all four instantiations):
// (kBlockM + 2 * kBlockN) * head_dim * 2 bytes.
int tile_smem_bytes(FlashTile t) {
  switch (t) {
    case FlashTile::kHdim128_128x64: return (128 + 2 * 64) * 128 * 2;  //  64 KiB
    case FlashTile::kHdim128_128x32: return (128 + 2 * 32) * 128 * 2;  //  48 KiB
    case FlashTile::kHdim256_128x64: return (128 + 2 * 64) * 256 * 2;  // 128 KiB
    case FlashTile::kHdim256_64x64: return (64 + 2 * 64) * 256 * 2;    //  96 KiB
    default: return 0;
  }
}

std::atomic<FlashTile> g_tile_override{FlashTile::kAuto};

// The kernels use cp.async and bf16 mma, which need sm_80+; on older parts they would compile to empty
// bodies (see MI_FLASH_PARAM_MODIFIER) and silently leave `out` unwritten. Checked once per device.
void require_sm80(int device) {
  static std::array<std::atomic<bool>, 64> ok{};
  if (device >= 0 && device < static_cast<int>(ok.size()) && ok[device].load()) return;
  int major = 0, minor = 0;
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major < 8) {
    throw std::runtime_error("flash_fwd: requires compute capability >= 8.0, device " + std::to_string(device) +
                             " is " + std::to_string(major) + "." + std::to_string(minor));
  }
  if (device >= 0 && device < static_cast<int>(ok.size())) ok[device].store(true);
}

}  // namespace

std::size_t flash_fwd_workspace_bytes(std::int64_t total_q, int num_heads) {
  return static_cast<std::size_t>(total_q) * static_cast<std::size_t>(num_heads) * sizeof(float);
}

void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream) {
  check_qkvo(q, "q");
  check_qkvo(k, "k");
  check_qkvo(v, "v");
  check_qkvo(out, "out");
  check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
  check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
  const std::int64_t total_q = q.dim(0), heads = q.dim(1), head_dim = q.dim(2);
  const std::int64_t kv_heads = k.dim(1);
  require(k.dim(2) == head_dim && v.dim(0) == k.dim(0) && v.dim(1) == kv_heads && v.dim(2) == head_dim,
          "k and v must both be [total_k, num_kv_heads, head_dim]");
  require(out.dim(0) == total_q && out.dim(1) == heads && out.dim(2) == head_dim, "out must have q's shape");
  require(kv_heads > 0 && heads % kv_heads == 0, "num_heads must be a multiple of num_kv_heads");
  require(cu_seqlens_k.dim(0) == cu_seqlens_q.dim(0), "cu_seqlens_q and cu_seqlens_k must have equal batch");
  require(max_seqlen > 0, "max_seqlen must be positive");
  require(std::isfinite(scale), "scale must be finite");
  // The kernels index rows with 32-bit offsets (uint32 sequence starts, int seqlens).
  require(total_q <= INT32_MAX && k.dim(0) <= INT32_MAX, "too many tokens");
  require(workspace_bytes >= flash_fwd_workspace_bytes(total_q, static_cast<int>(heads)) &&
              (workspace != nullptr || total_q == 0) && reinterpret_cast<std::uintptr_t>(workspace) % 4 == 0,
          "workspace too small or misaligned");
  require((head_dim == 128 && !causal) || (head_dim == 256 && causal),
          "unsupported configuration head_dim=" + std::to_string(head_dim) + (causal ? " causal" : " non-causal") +
              " (supported: 128 non-causal, 256 causal)");
  if (varlen_debug_checks()) {
    validate_cu_seqlens(cu_seqlens_q, max_seqlen, total_q, stream, "flash_fwd(cu_seqlens_q)");
    validate_cu_seqlens(cu_seqlens_k, max_seqlen, k.dim(0), stream, "flash_fwd(cu_seqlens_k)");
  }
  if (total_q == 0) return;

  // Mirrors mha_varlen_fwd + set_params_fprop (flash_api.cpp, v2.8.3) for: no dropout, no softcap,
  // no ALiBi, no paged KV, no window except causal (left = max_seqlen, right = 0).
  flash::Flash_fwd_params p{};
  p.is_bf16 = true;
  p.q_ptr = q.data;
  p.k_ptr = k.data;
  p.v_ptr = v.data;
  p.o_ptr = out.data;
  p.q_row_stride = q.stride[0];
  p.k_row_stride = k.stride[0];
  p.v_row_stride = v.stride[0];
  p.o_row_stride = out.stride[0];
  p.q_head_stride = q.stride[1];
  p.k_head_stride = k.stride[1];
  p.v_head_stride = v.stride[1];
  p.o_head_stride = out.stride[1];
  p.cu_seqlens_q = cu_seqlens_q.ptr<int>();
  p.cu_seqlens_k = cu_seqlens_k.ptr<int>();
  p.softmax_lse_ptr = workspace;
  p.b = static_cast<int>(cu_seqlens_q.dim(0) - 1);
  p.h = static_cast<int>(heads);
  p.h_k = static_cast<int>(kv_heads);
  p.h_h_k_ratio = static_cast<int>(heads / kv_heads);
  p.seqlen_q = max_seqlen;
  p.seqlen_k = max_seqlen;
  p.seqlen_q_rounded = (max_seqlen + 127) / 128 * 128;
  p.seqlen_k_rounded = p.seqlen_q_rounded;
  p.d = static_cast<int>(head_dim);
  p.d_rounded = p.d;
  p.total_q = static_cast<int>(total_q);
  p.scale_softmax = scale;
  // In double then rounded, as upstream (float * M_LOG2E): keeps the exp2 argument bit-identical.
  p.scale_softmax_log2 = static_cast<float>(static_cast<double>(scale) * kLog2e);
  p.p_dropout = 1.f;  // keep probability
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.f;
  p.scale_softmax_rp_dropout = scale;
  p.is_causal = causal;
  p.window_size_left = causal ? max_seqlen : -1;
  p.window_size_right = causal ? 0 : -1;
  p.is_seqlens_k_cumulative = true;
  p.unpadded_lse = true;

  // Tile choice as upstream run_mha_fwd_hdim128/256, keyed on shared memory instead of the compute
  // capability: a 128x64 hdim-128 tile takes 64 KiB, so on 100 KiB/SM parts (sm_86/89/120) the 48 KiB
  // 128x32 tile fits two CTAs per SM. Upstream picks 128x32 for sm_8x (x > 0) only; sm_120 is the one
  // difference.
  int device = 0, smem_per_sm = 0, smem_per_block = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  require_sm80(device);
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  const FlashTile forced = g_tile_override.load();
  if (forced != FlashTile::kAuto) {
    // Test hook: run exactly this instantiation (must match head_dim and fit the device's opt-in smem).
    const bool is128 = forced == FlashTile::kHdim128_128x64 || forced == FlashTile::kHdim128_128x32;
    require(is128 == (head_dim == 128), "forced tile does not match head_dim " + std::to_string(head_dim));
    require(smem_per_block >= tile_smem_bytes(forced),
            "forced tile needs " + std::to_string(tile_smem_bytes(forced)) + " bytes of shared memory per block, device allows " +
                std::to_string(smem_per_block));
    switch (forced) {
      case FlashTile::kHdim128_128x64: detail::run_fwd_hdim128_128x64(p, stream); break;
      case FlashTile::kHdim128_128x32: detail::run_fwd_hdim128_128x32(p, stream); break;
      case FlashTile::kHdim256_128x64: detail::run_fwd_hdim256_causal_128x64(p, stream); break;
      default: detail::run_fwd_hdim256_causal_64x64(p, stream); break;
    }
    return;
  }
  if (head_dim == 128) {
    if (smem_per_sm >= 2 * 64 * 1024) {
      detail::run_fwd_hdim128_128x64(p, stream);
    } else {
      detail::run_fwd_hdim128_128x32(p, stream);
    }
  } else {
    // Upstream: 128x64 (128 KiB) on A100-class parts, 64x64 (96 KiB) elsewhere (incl. H100 for 2 CTAs/SM).
    if (smem_per_block >= 2 * 256 * (128 + 2 * 64) && smem_per_sm < 4 * 256 * (64 + 2 * 64)) {
      detail::run_fwd_hdim256_causal_128x64(p, stream);
    } else {
      detail::run_fwd_hdim256_causal_64x64(p, stream);
    }
  }
}

void flash_fwd_set_tile_override_for_testing(FlashTile tile) { g_tile_override.store(tile); }

}  // namespace mi::attention
