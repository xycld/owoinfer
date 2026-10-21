// kernels/basic norms, modulation, RoPE, SiTU and elementwise ops: bit-exact against torch eager
// vectors from tools/reference/gen_basic_vectors.py (real-magnitude DiT/TE activations and weights).
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "util/npy.h"

namespace {

namespace k = mi::kernels;
using mi::DType;
using mi::Tensor;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/basic";

mi::npy::Array load(const std::string& name) { return mi::npy::load(kData / (name + ".npy")); }

// A device copy of an .npy array (bf16 stored as u2 bit patterns) and a contiguous view of it.
struct Dev {
  mi::DeviceBuffer buf;
  Tensor t;
};

Dev upload(const mi::npy::Array& a, DType dtype) {
  Dev d{mi::DeviceBuffer(a.data.size()), {}};
  d.buf.upload(a.data.data(), a.data.size());
  d.t = Tensor::view(d.buf.get(), dtype, mi::Device::kCUDA, a.shape.data(), static_cast<int>(a.shape.size()));
  return d;
}
Dev upload_bf16(const std::string& name) { return upload(load(name), DType::kBF16); }

Dev empty_like(const Tensor& t, DType dtype) {
  Dev d{mi::DeviceBuffer(static_cast<std::size_t>(t.numel()) * mi::dtype_size(dtype)), {}};
  d.t = Tensor::view(d.buf.get(), dtype, mi::Device::kCUDA, t.shape.data(), t.rank);
  return d;
}

// Downloads a (possibly strided) bf16 view in logical order.
std::vector<std::uint16_t> download(const Tensor& t) {
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::int64_t span = 1;
  for (int i = 0; i < t.rank; ++i) span += (t.shape[static_cast<std::size_t>(i)] - 1) * t.stride[static_cast<std::size_t>(i)];
  std::vector<std::uint16_t> raw(static_cast<std::size_t>(span));
  MI_CUDA_CHECK(cudaMemcpy(raw.data(), t.data, raw.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<std::uint16_t> out(static_cast<std::size_t>(t.numel()));
  for (std::int64_t i = 0; i < t.numel(); ++i) {
    std::int64_t rem = i, off = 0;
    for (int d = t.rank - 1; d >= 0; --d) {
      const auto u = static_cast<std::size_t>(d);
      off += (rem % t.shape[u]) * t.stride[u];
      rem /= t.shape[u];
    }
    out[static_cast<std::size_t>(i)] = raw[static_cast<std::size_t>(off)];
  }
  return out;
}

int bf16_ordered(std::uint16_t b) { return (b & 0x8000) ? -static_cast<int>(b & 0x7fff) : static_cast<int>(b); }

// Requires bit-identical bf16 output; on mismatch reports the count and max ULP distance.
void expect_bits(const std::vector<std::uint16_t>& got, const mi::npy::Array& want_arr, const std::string& what) {
  const std::vector<std::uint16_t> want = want_arr.to_vector<std::uint16_t>();
  ASSERT_EQ(got.size(), want.size()) << what;
  std::size_t mismatches = 0, first = 0;
  int max_ulp = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] == want[i]) continue;
    if (mismatches++ == 0) first = i;
    max_ulp = std::max(max_ulp, std::abs(bf16_ordered(got[i]) - bf16_ordered(want[i])));
  }
  EXPECT_EQ(mismatches, 0u) << what << ": " << mismatches << " of " << got.size() << " differ, max " << max_ulp
                            << " bf16 ULP, first at " << first << " (got 0x" << std::hex << got[first] << " want 0x"
                            << want[first] << ")";
}
void expect_bits(const Tensor& got, const std::string& want, const std::string& what = "") {
  expect_bits(download(got), load(want), what.empty() ? want : what);
}

// A copy of contiguous `src` whose first `lead` dims are rows, each row placed at column `col0` of a wider
// buffer with `pitch` elements per row: the strided views a fused projection gives (q inside qkv).
struct Strided {
  mi::DeviceBuffer buf;
  Tensor t;
};
Strided embed(const Tensor& src, int lead, std::int64_t pitch, std::int64_t col0) {
  std::int64_t rows = 1;
  for (int i = 0; i < lead; ++i) rows *= src.shape[static_cast<std::size_t>(i)];
  const std::int64_t width = src.numel() / rows;
  Strided s{mi::DeviceBuffer(static_cast<std::size_t>(rows * pitch) * 2), {}};
  MI_CUDA_CHECK(cudaMemset(s.buf.get(), 0xff, s.buf.size()));
  MI_CUDA_CHECK(cudaMemcpy2D(static_cast<char*>(s.buf.get()) + col0 * 2, static_cast<std::size_t>(pitch) * 2, src.data,
                             static_cast<std::size_t>(width) * 2, static_cast<std::size_t>(width) * 2,
                             static_cast<std::size_t>(rows), cudaMemcpyDeviceToDevice));
  s.t = src;
  s.t.data = static_cast<char*>(s.buf.get()) + col0 * 2;
  std::int64_t stride = pitch;
  for (int i = lead - 1; i >= 0; --i) {
    s.t.stride[static_cast<std::size_t>(i)] = stride;
    stride *= src.shape[static_cast<std::size_t>(i)];
  }
  return s;
}

}  // namespace

// ------------------------------------------------------------------------------------------- norms

TEST(GpuKernelsBasic, RmsNormMoss) {
  Dev x = upload_bf16("moss_norm_x"), w = upload_bf16("moss_norm_w");
  Dev out = empty_like(x.t, DType::kBF16);
  k::rms_norm_moss(x.t, w.t, out.t, nullptr);
  expect_bits(out.t, "moss_norm_out");
  // Strided input and output rows (a 3072-wide slice of 9216-wide rows).
  Strided xs = embed(x.t, 1, 9216, 3072), os = embed(out.t, 1, 6144, 3072);
  MI_CUDA_CHECK(cudaMemset(os.buf.get(), 0, os.buf.size()));
  k::rms_norm_moss(xs.t, w.t, os.t, nullptr);
  expect_bits(os.t, "moss_norm_out", "moss_norm_out (strided)");
  // In place.
  k::rms_norm_moss(x.t, w.t, x.t, nullptr);
  expect_bits(x.t, "moss_norm_out", "moss_norm_out (in place)");
}

TEST(GpuKernelsBasic, RmsNormModulateMoss) {
  Dev x = upload_bf16("moss_norm_x"), w = upload_bf16("moss_norm_w");
  Dev scale = upload_bf16("moss_mod_scale"), shift = upload_bf16("moss_mod_shift");
  Dev out = empty_like(x.t, DType::kBF16);
  k::rms_norm_modulate_moss(x.t, w.t, scale.t, shift.t, out.t, nullptr);
  expect_bits(out.t, "moss_mod_out");
}

TEST(GpuKernelsBasic, RmsNormMossPerHead) {
  Dev x = upload_bf16("moss_qnorm_x"), w = upload_bf16("moss_qnorm_w");  // [32, 24, 128]
  Dev out = empty_like(x.t, DType::kBF16);
  k::rms_norm_moss(x.t, w.t, out.t, nullptr);
  expect_bits(out.t, "moss_qnorm_out");
  // q heads inside a fused [tokens, 3*3072] qkv row (text refiner layout).
  Strided xs = embed(x.t, 1, 9216, 0);
  k::rms_norm_moss(xs.t, w.t, out.t, nullptr);
  expect_bits(out.t, "moss_qnorm_out", "moss_qnorm_out (strided heads)");
}

TEST(GpuKernelsBasic, RmsNormQwenAllReduceShapes) {
  Dev w = upload_bf16("qwen_norm_w");
  for (int r : {1, 2, 5, 9, 16, 17}) {
    Dev x = upload_bf16("qwen_norm_x_r" + std::to_string(r));
    Dev out = empty_like(x.t, DType::kBF16);
    k::rms_norm_qwen(x.t, w.t, out.t, nullptr);
    expect_bits(out.t, "qwen_norm_out_r" + std::to_string(r));
  }
}

TEST(GpuKernelsBasic, RmsNormQwenPerHead) {
  // q: [9, 16, 256] read through Qwen3_5Attention's per-head [q | gate] chunk (head stride 512).
  Dev q = upload_bf16("qwen_qnorm_x"), wq = upload_bf16("qwen_qnorm_w");
  Strided qs = embed(q.t, 2, 512, 0);
  Dev out = empty_like(q.t, DType::kBF16);
  k::rms_norm_qwen(qs.t, wq.t, out.t, nullptr);
  expect_bits(out.t, "qwen_qnorm_out");
  // k: [2, 4, 256] = 8 rows, where torch uses a 64 x 8 reduce block.
  Dev kx = upload_bf16("qwen_knorm_x"), wk = upload_bf16("qwen_knorm_w");
  Dev kout = empty_like(kx.t, DType::kBF16);
  k::rms_norm_qwen(kx.t, wk.t, kout.t, nullptr);
  expect_bits(kout.t, "qwen_knorm_out");
}

TEST(GpuKernelsBasic, RmsNormGatedQwen) {
  Dev x = upload_bf16("qwen_gnorm_x"), g = upload_bf16("qwen_gnorm_gate"), w = upload_bf16("qwen_gnorm_w");
  Dev out = empty_like(x.t, DType::kBF16);
  k::rms_norm_gated_qwen(x.t, g.t, w.t, out.t, nullptr);
  expect_bits(out.t, "qwen_gnorm_out");
}

TEST(GpuKernelsBasic, RmsNormRejectsBadArguments) {
  Dev x = upload_bf16("moss_norm_x"), w = upload_bf16("moss_qnorm_w");
  Dev out = empty_like(x.t, DType::kBF16);
  EXPECT_THROW(k::rms_norm_moss(x.t, w.t, out.t, nullptr), std::invalid_argument);  // weight [128] vs D 3072
  Dev f32 = empty_like(x.t, DType::kF32);
  Dev w3072 = upload_bf16("moss_norm_w");
  EXPECT_THROW(k::rms_norm_moss(x.t, w3072.t, f32.t, nullptr), std::invalid_argument);
}

// ------------------------------------------------------------------------------------- elementwise

TEST(GpuKernelsBasic, SituGlu) {
  Dev ug = upload_bf16("situ_up_gate");  // [16, 16384] = [hidden | gate]
  const std::int64_t rows = ug.t.dim(0), f = ug.t.dim(1) / 2;
  mi::DeviceBuffer obuf(static_cast<std::size_t>(rows * f) * 2);
  Tensor out = Tensor::view(obuf.get(), DType::kBF16, mi::Device::kCUDA, {rows, f});
  k::situ_glu(ug.t, out, nullptr);
  expect_bits(out, "situ_out");
  // Separate-tensor form on strided views of the same buffer.
  Tensor hidden = Tensor::view(ug.t.data, DType::kBF16, mi::Device::kCUDA, {rows, f});
  hidden.stride[0] = 2 * f;
  Tensor gate = hidden;
  gate.data = static_cast<char*>(ug.t.data) + f * 2;
  MI_CUDA_CHECK(cudaMemset(obuf.get(), 0, obuf.size()));
  k::situ(hidden, gate, out, nullptr);
  expect_bits(out, "situ_out", "situ_out (separate)");
}

TEST(GpuKernelsBasic, GatedResidualAndAdd) {
  Dev x = upload_bf16("moss_norm_x"), y = upload_bf16("gres_y"), g = upload_bf16("gres_gate");
  Dev out = empty_like(x.t, DType::kBF16);
  k::gated_residual(x.t, y.t, g.t, out.t, nullptr);
  expect_bits(out.t, "gres_out");
  k::residual_add(x.t, y.t, out.t, nullptr);
  expect_bits(out.t, "add_out");
  k::gated_residual(x.t, y.t, g.t, x.t, nullptr);  // in place on the residual stream
  expect_bits(x.t, "gres_out", "gres_out (in place)");
}

TEST(GpuKernelsBasic, SiluFamily) {
  Dev a = upload_bf16("ew_a"), b = upload_bf16("ew_b");
  Dev out = empty_like(a.t, DType::kBF16);
  k::silu(a.t, out.t, nullptr);
  expect_bits(out.t, "silu_out");
  k::silu_mul(a.t, b.t, out.t, nullptr);
  expect_bits(out.t, "silu_mul_out");
  k::sigmoid_mul(b.t, a.t, out.t, nullptr);
  expect_bits(out.t, "sigmoid_mul_out");
}

TEST(GpuKernelsBasic, Cast) {
  Dev in = upload(load("cast_f32"), DType::kF32);
  Dev bf = empty_like(in.t, DType::kBF16);
  k::cast(in.t, bf.t, nullptr);
  expect_bits(bf.t, "cast_bf16");
  Dev back = empty_like(in.t, DType::kF32);
  k::cast(bf.t, back.t, nullptr);
  std::vector<float> widened(static_cast<std::size_t>(in.t.numel()));
  back.buf.download(widened.data(), widened.size() * 4);
  const std::vector<std::uint16_t> want = load("cast_bf16").to_vector<std::uint16_t>();
  for (std::size_t i = 0; i < widened.size(); ++i) {
    ASSERT_EQ(std::bit_cast<std::uint32_t>(widened[i]), static_cast<std::uint32_t>(want[i]) << 16) << i;
  }
}

// -------------------------------------------------------------------------------------------- RoPE

TEST(GpuKernelsBasic, RopeFrequencyConstants) {
  const std::vector<float> bands = load("moss_rope_bands").to_vector<float>();
  ASSERT_EQ(bands.size(), k::kMossRopeBandBits.size());
  for (std::size_t i = 0; i < bands.size(); ++i) EXPECT_EQ(std::bit_cast<std::uint32_t>(bands[i]), k::kMossRopeBandBits[i]) << i;
  const std::vector<float> inv = load("qwen_rope_inv_freq").to_vector<float>();
  ASSERT_EQ(inv.size(), k::kQwenRopeInvFreqBits.size());
  for (std::size_t i = 0; i < inv.size(); ++i) EXPECT_EQ(std::bit_cast<std::uint32_t>(inv[i]), k::kQwenRopeInvFreqBits[i]) << i;
}

TEST(GpuKernelsBasic, MossRopeTable) {
  for (int c = 0; c < 4; ++c) {
    const std::string p = "moss_rope_case" + std::to_string(c);
    const std::vector<std::int64_t> hwt = load(p + "_hwt").to_vector<std::int64_t>();
    const std::int64_t n = hwt[0] * hwt[1] + hwt[2];
    mi::DeviceBuffer buf(static_cast<std::size_t>(n) * 128 * 2);
    Tensor table = Tensor::view(buf.get(), DType::kBF16, mi::Device::kCUDA, {n, 128});
    k::moss_rope_table(table, static_cast<int>(hwt[0]), static_cast<int>(hwt[1]), static_cast<int>(hwt[2]), nullptr);
    const std::vector<std::uint16_t> all = download(table);
    std::vector<std::uint16_t> picked;
    for (std::int64_t r : load(p + "_rows").to_vector<std::int64_t>()) {
      picked.insert(picked.end(), all.begin() + r * 128, all.begin() + (r + 1) * 128);
    }
    expect_bits(picked, load(p + "_vals"), p + " (" + std::to_string(hwt[0]) + "x" + std::to_string(hwt[1]) + ", T=" +
                                               std::to_string(hwt[2]) + ")");
  }
}

TEST(GpuKernelsBasic, MossRopeApply) {
  Dev q = upload_bf16("moss_rope_q"), tab = upload_bf16("moss_rope_tab");  // [L, 24, 128], [L, 128]
  Dev out = empty_like(q.t, DType::kBF16);
  k::moss_rope_apply(q.t, tab.t, out.t, nullptr);
  expect_bits(out.t, "moss_rope_out");
  // Batched leading dim [2, L, heads, 128] sharing one table, and in place.
  mi::DeviceBuffer two(q.buf.size() * 2);
  MI_CUDA_CHECK(cudaMemcpy(two.get(), q.t.data, q.buf.size(), cudaMemcpyDeviceToDevice));
  MI_CUDA_CHECK(cudaMemcpy(static_cast<char*>(two.get()) + q.buf.size(), q.t.data, q.buf.size(), cudaMemcpyDeviceToDevice));
  Tensor qb = Tensor::view(two.get(), DType::kBF16, mi::Device::kCUDA, {2, q.t.dim(0), 24, 128});
  k::moss_rope_apply(qb, tab.t, qb, nullptr);
  expect_bits(qb.slice0(1, 2), "moss_rope_out", "moss_rope_out (batch 1, in place)");
  EXPECT_THROW(k::moss_rope_apply(q.t, tab.t.slice0(0, q.t.dim(0) - 1), out.t, nullptr), std::invalid_argument);
}

// GPU table -> text-refiner slice (P:251) -> apply, against torch's _rotate on the same text tokens.
TEST(GpuKernelsBasic, MossRopeTextRefinerSlice) {
  const std::vector<std::int64_t> hwt = load("moss_rope_case0_hwt").to_vector<std::int64_t>();  // the q's sequence
  const std::int64_t hw = hwt[0] * hwt[1], t = hwt[2];
  mi::DeviceBuffer buf(static_cast<std::size_t>(hw + t) * 128 * 2);
  Tensor table = Tensor::view(buf.get(), DType::kBF16, mi::Device::kCUDA, {hw + t, 128});
  k::moss_rope_table(table, static_cast<int>(hwt[0]), static_cast<int>(hwt[1]), static_cast<int>(t), nullptr);

  // Scatter the stored text-token q rows to their text positions in a [T, 24, 128] tensor.
  const std::vector<std::int64_t> rows = load("moss_rope_token_rows").to_vector<std::int64_t>();
  const mi::npy::Array q = load("moss_rope_q"), want = load("moss_rope_out");
  constexpr std::size_t kRow = 24 * 128;
  std::vector<std::uint16_t> qt(static_cast<std::size_t>(t) * kRow, 0), want_text, picked;
  const auto* qbits = reinterpret_cast<const std::uint16_t*>(q.data.data());
  const auto* wbits = reinterpret_cast<const std::uint16_t*>(want.data.data());
  std::vector<std::int64_t> positions;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] < hw) continue;
    positions.push_back(rows[i] - hw);
    std::copy(qbits + i * kRow, qbits + (i + 1) * kRow, qt.begin() + (rows[i] - hw) * static_cast<std::int64_t>(kRow));
    want_text.insert(want_text.end(), wbits + i * kRow, wbits + (i + 1) * kRow);
  }
  ASSERT_FALSE(positions.empty());
  mi::DeviceBuffer xb(qt.size() * 2);
  xb.upload(qt.data(), qt.size() * 2);
  Tensor x = Tensor::view(xb.get(), DType::kBF16, mi::Device::kCUDA, {t, 24, 128});
  k::moss_rope_apply(x, table.slice0(hw, hw + t), x, nullptr);
  const std::vector<std::uint16_t> got = download(x);
  for (std::int64_t p : positions) picked.insert(picked.end(), got.begin() + p * kRow, got.begin() + (p + 1) * kRow);
  mi::npy::Array want_arr;
  want_arr.dtype = "u2";
  want_arr.shape = {static_cast<std::int64_t>(want_text.size())};
  want_arr.data.resize(want_text.size() * 2);
  std::memcpy(want_arr.data.data(), want_text.data(), want_arr.data.size());
  expect_bits(picked, want_arr, "moss_rope text-refiner slice");
}

TEST(GpuKernelsBasic, QwenRopeTableAndApply) {
  const mi::npy::Array want_table = load("qwen_rope_table");  // [4096, 64]
  const std::int64_t s = want_table.shape[0];
  mi::DeviceBuffer buf(static_cast<std::size_t>(s) * 64 * 2);
  Tensor table = Tensor::view(buf.get(), DType::kBF16, mi::Device::kCUDA, {s, 64});
  k::qwen_rope_table(table, nullptr);
  expect_bits(download(table), want_table, "qwen_rope_table");

  Dev q = upload_bf16("qwen_rope_q"), kk = upload_bf16("qwen_rope_k");  // [24, 16, 256], [24, 4, 256]
  Tensor tab = table.slice0(0, q.t.dim(0));
  Dev qo = empty_like(q.t, DType::kBF16);
  k::qwen_rope_apply(q.t, tab, qo.t, nullptr);
  expect_bits(qo.t, "qwen_rope_q_out");
  k::qwen_rope_apply(kk.t, tab, kk.t, nullptr);
  expect_bits(kk.t, "qwen_rope_k_out", "qwen_rope_k_out (in place)");
}
