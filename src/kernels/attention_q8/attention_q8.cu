#include "kernels/attention_q8/attention_q8.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention_q8/attention_q8_testing.h"
#include "kernels/attention_q8/q8_kernels.h"

namespace mi::attention_q8 {
namespace {

using detail::kCtaK;
using detail::kCtaQ;
using detail::kHeadDim;
using detail::kStatsChunk;

constexpr std::size_t kAlign = 256;
constexpr float kFp16Max = 65504.f;
constexpr std::int64_t kMaxGridYz = 65535;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("attention_q8: " + what);
}

std::size_t align_up(std::size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

std::atomic<cudaEvent_t> g_prepass_event{nullptr};

bool is_fp8(Variant v) { return v == Variant::kInt8Fp8 || v == Variant::kInt8Fp8F32Acc; }
int warp_q(Variant v) { return v == Variant::kInt8Fp16 ? 16 : 32; }

// Byte layout of the workspace (see attention_q8.h); offsets from the 256-byte aligned base.
struct Layout {
  std::size_t q_i8, k_i8, v, q_scale, k_scale, k_mean, v_scale, v_recp, part_sum, part_amax, total;
  std::int64_t padded_k;
  int q_blocks, k_blocks, chunks;
};

Layout layout(Variant var, std::int64_t total_q, std::int64_t total_k, int batch, int max_seqlen, int heads, int kv_heads) {
  Layout l{};
  const auto n = [](std::int64_t v) { return static_cast<std::size_t>(v); };
  l.padded_k = (total_k + std::int64_t{kCtaK} * batch + kCtaK - 1) / kCtaK * kCtaK;
  l.q_blocks = (max_seqlen + kCtaQ - 1) / kCtaQ * (kCtaQ / warp_q(var));
  l.k_blocks = (max_seqlen + kCtaK - 1) / kCtaK;
  l.chunks = (max_seqlen + kStatsChunk - 1) / kStatsChunk;
  std::size_t at = 0;
  const auto take = [&](std::size_t bytes) {
    const std::size_t off = at;
    at += align_up(bytes);
    return off;
  };
  l.q_i8 = take(n(total_q) * n(heads) * kHeadDim);
  l.k_i8 = take(n(total_k) * n(kv_heads) * kHeadDim);
  l.v = take(is_fp8(var) ? n(kv_heads) * kHeadDim * n(l.padded_k) : n(total_k) * n(kv_heads) * kHeadDim * 2);
  l.q_scale = take(n(batch) * n(heads) * n(l.q_blocks) * 4);
  l.k_scale = take(n(batch) * n(kv_heads) * n(l.k_blocks) * 4);
  l.k_mean = take(n(batch) * n(kv_heads) * kHeadDim * 4);
  // The V statistics also exist for the fp16 variants: varlen_debug_checks() measures max|V| there.
  l.v_scale = take(n(batch) * n(kv_heads) * kHeadDim * 4);
  l.v_recp = take(n(batch) * n(kv_heads) * kHeadDim * 4);
  l.part_sum = take(n(batch) * n(kv_heads) * n(l.chunks) * kHeadDim * 4);
  l.part_amax = take(n(batch) * n(kv_heads) * n(l.chunks) * kHeadDim * 4);
  l.total = at;
  return l;
}

// 128-bit loads and stores need 16-byte aligned rows and heads (as flash_fwd).
void check_qkvo(const Tensor& t, const char* name) {
  require(t.device == Device::kCUDA && t.dtype == DType::kBF16 && t.rank == 3,
          std::string(name) + " must be a bf16 [tokens, heads, head_dim] CUDA tensor, got " + t.describe());
  require(t.stride[2] == 1, std::string(name) + " head_dim must be contiguous");
  require(t.stride[0] % 8 == 0 && t.stride[1] % 8 == 0 && reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0,
          std::string(name) + " needs 16-byte aligned base and row/head strides that are multiples of 8");
}

// Largest element offset a [tokens, heads, 128] view touches (0 when empty).
std::int64_t max_offset(const Tensor& t) {
  if (t.dim(0) == 0 || t.dim(1) == 0) return 0;
  return (t.dim(0) - 1) * t.stride[0] + (t.dim(1) - 1) * t.stride[1] + t.dim(2) - 1;
}

void check_cu_seqlens(const Tensor& t, const char* name) {
  require(t.device == Device::kCUDA && t.dtype == DType::kI32 && t.rank == 1 && t.dim(0) >= 2 && t.stride[0] == 1,
          std::string(name) + " must be a contiguous int32 [batch + 1] CUDA tensor, got " + t.describe());
}

// Why variant `v` cannot run on device `device` (see attention_q8.h), or kOk. Cached per (device, variant).
enum class Support : int { kUnknown = 0, kOk, kOldDevice, kOldCompiler, kNoImage };
constexpr int kMaxDevices = 64;

int device_cc(int device) {
  int major = 0, minor = 0;
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return major * 10 + minor;
}

int need_cc(Variant v) { return is_fp8(v) ? 89 : 80; }

// The architecture of the variant's kernel image as loaded on the current device (support() is only asked about it).
int image_cc(Variant v) {
  switch (v) {
    case Variant::kInt8Fp16: return detail::int8_fp16_kernel_arch(false);
    case Variant::kInt8Fp16F32Acc: return detail::int8_fp16_kernel_arch(true);
    case Variant::kInt8Fp8: return detail::int8_fp8_kernel_arch(false);
    case Variant::kInt8Fp8F32Acc: return detail::int8_fp8_kernel_arch(true);
  }
  return 0;
}

Support support(int device, Variant v) {
  static std::array<std::array<std::atomic<int>, 4>, kMaxDevices> cache{};
  const bool cached = device >= 0 && device < kMaxDevices;
  if (cached) {
    if (const int c = cache[device][static_cast<int>(v)].load(); c != 0) return static_cast<Support>(c);
  }
  Support r = Support::kOk;
  if (device_cc(device) < need_cc(v)) {
    r = Support::kOldDevice;
  } else if (v == Variant::kInt8Fp8 && !detail::int8_fp8_f16_accum_compiled()) {
    r = Support::kOldCompiler;
  } else if (image_cc(v) < need_cc(v)) {
    r = Support::kNoImage;
  }
  if (cached) cache[device][static_cast<int>(v)].store(static_cast<int>(r));
  return r;
}

std::string cc_string(int cc) { return std::to_string(cc / 10) + "." + std::to_string(cc % 10); }

void require_supported(int device, Variant v) {
  const std::string who = std::string("attention_q8: ") + variant_name(v);
  const std::string need = cc_string(need_cc(v));
  switch (support(device, v)) {
    case Support::kOk:
    case Support::kUnknown: return;
    case Support::kOldDevice:
      throw std::runtime_error(who + " needs compute capability >= " + need + ", device " + std::to_string(device) +
                               " is " + cc_string(device_cc(device)));
    case Support::kOldCompiler:
      throw std::runtime_error(who + " needs nvcc >= 12.8 at build time (fp16-accumulating e4m3 MMA); use int8_fp8_f32acc");
    case Support::kNoImage:
      throw std::runtime_error(who + " needs a kernel image compiled for compute capability >= " + need +
                               " (add " + std::to_string(need_cc(v)) + " or newer to CMAKE_CUDA_ARCHITECTURES); device " + std::to_string(device) +
                               " loads one for " + cc_string(image_cc(v)));
  }
}

// e4m3 round-to-nearest-even with saturation to 448 (cvt.rn.satfinite.e4m3x2.f32) of a finite x >= 0.
double e4m3_rn(double x) {
  if (x >= 448.0) return 448.0;
  int e = 0;
  std::frexp(x, &e);  // x = m * 2^e, m in [0.5, 1)
  const double step = std::ldexp(1.0, std::max(e - 1, -6) - 3);  // 3 mantissa bits; subnormals below 2^-6
  return std::min(448.0, std::nearbyint(x / step) * step);
}

}  // namespace

bool supported(Variant v) {
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  return support(device, v) == Support::kOk;
}

const char* variant_name(Variant v) {
  switch (v) {
    case Variant::kInt8Fp16: return "int8_fp16";
    case Variant::kInt8Fp16F32Acc: return "int8_fp16_f32acc";
    case Variant::kInt8Fp8: return "int8_fp8";
    case Variant::kInt8Fp8F32Acc: return "int8_fp8_f32acc";
  }
  return "?";
}

float effective_p_range(const Options& o) {
  if (!is_fp8(o.variant)) return 0.f;
  if (o.p_range > 0) return o.p_range;
  return o.variant == Variant::kInt8Fp8 ? 112.f : 448.f;
}

float effective_v_range(const Options& o) {
  if (!is_fp8(o.variant)) return 0.f;
  if (o.v_range > 0) return o.v_range;
  return o.variant == Variant::kInt8Fp8 ? 4.5f : 448.f;
}

std::size_t attention_q8_workspace_bytes(const Options& o, std::int64_t total_q, std::int64_t total_k, int batch,
                                         int max_seqlen, int num_heads, int num_kv_heads) {
  return layout(o.variant, total_q, total_k, batch, max_seqlen, num_heads, num_kv_heads).total;
}

void attention_q8(const Options& o, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                  const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, int max_seqlen, float scale, void* workspace,
                  std::size_t workspace_bytes, cudaStream_t stream, const QkNormRope* norm_rope) {
  check_qkvo(q, "q");
  check_qkvo(k, "k");
  check_qkvo(v, "v");
  check_qkvo(out, "out");
  check_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
  check_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
  const std::int64_t total_q = q.dim(0), heads = q.dim(1), head_dim = q.dim(2);
  const std::int64_t total_k = k.dim(0), kv_heads = k.dim(1);
  require(head_dim == kHeadDim, "unsupported head_dim " + std::to_string(head_dim) + " (supported: 128)");
  require(k.dim(2) == head_dim && v.dim(0) == total_k && v.dim(1) == kv_heads && v.dim(2) == head_dim,
          "k and v must both be [total_k, num_kv_heads, head_dim]");
  require(out.dim(0) == total_q && out.dim(1) == heads && out.dim(2) == head_dim, "out must have q's shape");
  require(kv_heads > 0 && heads % kv_heads == 0, "num_heads must be a multiple of num_kv_heads");
  require(heads <= kMaxGridYz, "num_heads must be <= 65535 (grid y)");
  require(cu_seqlens_k.dim(0) == cu_seqlens_q.dim(0), "cu_seqlens_q and cu_seqlens_k must have equal batch");
  require(cu_seqlens_q.dim(0) - 1 <= kMaxGridYz, "batch must be <= 65535 (grid z)");
  require(max_seqlen > 0, "max_seqlen must be positive");
  require(std::isfinite(scale) && scale > 0, "scale must be finite and positive");
  if (norm_rope != nullptr) {
    const auto vec128 = [](const Tensor& t) {
      return t.dtype == DType::kBF16 && t.device == Device::kCUDA && t.rank == 1 && t.dim(0) == kHeadDim &&
             t.is_contiguous() && reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0;
    };
    const Tensor& tab = norm_rope->table;
    const std::int64_t tab_row = tab.rank == 2 && tab.dim(0) > 1 ? tab.stride[0] : kHeadDim;
    require(vec128(norm_rope->q_norm) && vec128(norm_rope->k_norm), "norm_rope: q_norm and k_norm must be contiguous "
                                                                     "16-byte aligned bf16 [128] device vectors");
    require(tab.dtype == DType::kBF16 && tab.device == Device::kCUDA && tab.rank == 2 && tab.dim(0) == total_q &&
                tab.dim(1) == kHeadDim && tab.stride[1] == 1 && tab_row % 8 == 0 &&
                reinterpret_cast<std::uintptr_t>(tab.data) % 16 == 0,
            "norm_rope: table must be a bf16 [total_q, 128] device tensor, 16-byte aligned rows");
    require(kv_heads == heads && cu_seqlens_k.data == cu_seqlens_q.data && k.stride[0] == q.stride[0],
            "norm_rope needs num_heads == num_kv_heads, one cu_seqlens tensor and q and k rows of one stride");
  }
  const int batch = static_cast<int>(cu_seqlens_q.dim(0) - 1);
  const Layout l = layout(o.variant, total_q, total_k, batch, max_seqlen, static_cast<int>(heads), static_cast<int>(kv_heads));
  // The vendored kernels index with 32-bit element offsets: unsigned into the int8/fp16/fp8 buffers and the bf16
  // output (through its strides), and the inputs are kept below INT32_MAX elements as well.
  const std::int64_t kU32 = std::int64_t{1} << 32;
  require(total_q * heads * kHeadDim < kU32 && total_k * kv_heads * kHeadDim < kU32 &&
              kv_heads * kHeadDim * l.padded_k < kU32 && max_offset(out) < kU32 && max_offset(q) < INT32_MAX &&
              max_offset(k) < INT32_MAX && max_offset(v) < INT32_MAX,
          "too many tokens");
  require(total_q == 0 || (reinterpret_cast<std::uintptr_t>(workspace) % kAlign == 0 && workspace != nullptr &&
                           workspace_bytes >= l.total),
          "workspace too small or not 256-byte aligned (need " + std::to_string(l.total) + " bytes)");
  const float p_range = effective_p_range(o), v_range = effective_v_range(o);
  // log2(p_range) truncated to 3 decimals: upstream's S_FP8_OFFSET 8.807 for 448 (the row max lands just below the
  // e4m3 max), 6.807 for 112. P's largest value is 2^p_offset, quantized to e4m3.
  const float p_offset =
      is_fp8(o.variant) ? static_cast<float>(std::floor(std::log2(static_cast<double>(p_range)) * 1000.0) / 1000.0) : 0.f;
  if (is_fp8(o.variant)) {
    require(std::isfinite(p_range) && p_range <= 448.f && std::isfinite(v_range) && v_range <= 448.f,
            "p_range and v_range must be in (0, 448] (e4m3 max)");
    const double p_max = e4m3_rn(std::exp2(static_cast<double>(p_offset))), v_max = e4m3_rn(v_range);
    require(o.variant != Variant::kInt8Fp8 || 64.0 * p_max * v_max <= kFp16Max,
            "64 * e4m3(2^p_offset) * e4m3(v_range) = 64 * " + std::to_string(p_max) + " * " + std::to_string(v_max) +
                " must be <= 65504 with fp16 accumulation");
  }
  if (varlen_debug_checks()) {
    validate_cu_seqlens(cu_seqlens_q, max_seqlen, total_q, stream, "attention_q8(cu_seqlens_q)");
    validate_cu_seqlens(cu_seqlens_k, max_seqlen, total_k, stream, "attention_q8(cu_seqlens_k)");
  }
  if (total_q == 0) return;
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  require_supported(device, o.variant);
  // Debug checks: max|V| against the fp16 PV limits (attention_q8.h), measured by the statistics pass. Skipped while
  // the stream is captured (the copy and sync would invalidate the capture), as validate_cu_seqlens.
  bool check_v = false;
  if (!is_fp8(o.variant) && varlen_debug_checks()) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    MI_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    check_v = capture == cudaStreamCaptureStatusNone;
  }

  auto* base = static_cast<std::byte*>(workspace);
  detail::PrepParams pp;
  pp.q = q.data;
  pp.k = k.data;
  pp.v = v.data;
  pp.q_row = q.stride[0];
  pp.q_head = q.stride[1];
  pp.k_row = k.stride[0];
  pp.k_head = k.stride[1];
  pp.v_row = v.stride[0];
  pp.v_head = v.stride[1];
  pp.cu_q = cu_seqlens_q.ptr<int>();
  pp.cu_k = cu_seqlens_k.ptr<int>();
  pp.batch = batch;
  pp.heads = static_cast<int>(heads);
  pp.kv_heads = static_cast<int>(kv_heads);
  pp.max_seqlen = max_seqlen;
  pp.warp_q = warp_q(o.variant);
  pp.fp8 = is_fp8(o.variant);
  pp.v_amax = check_v;
  pp.v_range = pp.fp8 ? v_range : 1.f;
  pp.padded_k = l.padded_k;
  pp.chunks = l.chunks;
  pp.q_i8 = reinterpret_cast<std::int8_t*>(base + l.q_i8);
  pp.k_i8 = reinterpret_cast<std::int8_t*>(base + l.k_i8);
  pp.v_out = base + l.v;
  pp.q_scale = reinterpret_cast<float*>(base + l.q_scale);
  pp.k_scale = reinterpret_cast<float*>(base + l.k_scale);
  pp.k_mean = reinterpret_cast<float*>(base + l.k_mean);
  pp.v_scale = reinterpret_cast<float*>(base + l.v_scale);
  pp.v_recp = reinterpret_cast<float*>(base + l.v_recp);
  pp.part_sum = reinterpret_cast<float*>(base + l.part_sum);
  pp.part_amax = reinterpret_cast<float*>(base + l.part_amax);
  if (norm_rope != nullptr) {
    // qk_norm_rope_moss's mean factor for a [total_q, heads, 128] head view: rows = total_q * heads.
    const std::int64_t rows = total_q * heads;
    pp.norm_rope = true;
    pp.q_norm = norm_rope->q_norm.data;
    pp.k_norm = norm_rope->k_norm.data;
    pp.table = norm_rope->table.data;
    pp.table_row = norm_rope->table.dim(0) > 1 ? norm_rope->table.stride[0] : kHeadDim;
    pp.norm_factor = static_cast<float>(rows) / static_cast<float>(rows * kHeadDim);
  }
  detail::preprocess(pp, stream);
  if (check_v) {
    std::vector<float> amax(static_cast<std::size_t>(batch) * kv_heads * kHeadDim);
    MI_CUDA_CHECK(cudaMemcpyAsync(amax.data(), pp.v_scale, amax.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    const float limit = o.variant == Variant::kInt8Fp16 ? kFp16Max / 64.f : kFp16Max;
    const float got = amax.empty() ? 0.f : *std::max_element(amax.begin(), amax.end());
    require(got <= limit, std::string(variant_name(o.variant)) + ": max|V| " + std::to_string(got) + " exceeds " +
                              std::to_string(limit) + " (fp16 PV overflow; use an fp8 variant)");
  }
  if (cudaEvent_t e = g_prepass_event.load()) MI_CUDA_CHECK(cudaEventRecord(e, stream));

  detail::AttnParams ap;
  ap.q = pp.q_i8;
  ap.k = pp.k_i8;
  ap.v = pp.v_out;
  ap.o = out.data;
  ap.q_scale = pp.q_scale;
  ap.k_scale = pp.k_scale;
  ap.v_scale = pp.v_scale;
  ap.cu_q = pp.cu_q;
  ap.cu_k = pp.cu_k;
  ap.batch = batch;
  ap.heads = static_cast<int>(heads);
  ap.kv_groups = static_cast<int>(heads / kv_heads);
  ap.max_seqlen = max_seqlen;
  ap.k_scale_blocks = static_cast<std::uint32_t>(l.k_blocks);
  ap.stride_seq_q = static_cast<std::uint32_t>(heads * kHeadDim);
  ap.stride_h_q = kHeadDim;
  ap.stride_seq_k = static_cast<std::uint32_t>(kv_heads * kHeadDim);
  ap.stride_h_k = kHeadDim;
  if (pp.fp8) {
    ap.stride_seq_v = static_cast<std::uint32_t>(l.padded_k);             // per channel row
    ap.stride_h_v = static_cast<std::uint32_t>(kHeadDim * l.padded_k);  // per kv head
  } else {
    ap.stride_seq_v = static_cast<std::uint32_t>(kv_heads * kHeadDim);
    ap.stride_h_v = kHeadDim;
  }
  ap.stride_seq_o = static_cast<std::uint32_t>(out.stride[0]);
  ap.stride_h_o = static_cast<std::uint32_t>(out.stride[1]);
  ap.sm_scale = scale;
  ap.p_offset = p_offset;
  switch (o.variant) {
    case Variant::kInt8Fp16: detail::run_int8_fp16(ap, false, stream); break;
    case Variant::kInt8Fp16F32Acc: detail::run_int8_fp16(ap, true, stream); break;
    case Variant::kInt8Fp8: detail::run_int8_fp8(ap, false, stream); break;
    case Variant::kInt8Fp8F32Acc: detail::run_int8_fp8(ap, true, stream); break;
  }
}

void set_prepass_event_for_testing(cudaEvent_t e) { g_prepass_event.store(e); }

}  // namespace mi::attention_q8
