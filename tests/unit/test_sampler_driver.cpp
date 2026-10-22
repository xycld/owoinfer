// Lane D (AC25): the sampler driver on the GPU against the k-diffusion family reference run as an oracle
// (tests/data/samplers): per-step updates and whole trajectories with identical model outputs, torch's CUDA randn,
// and the data-space sampler.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "pipeline/rng_torch_compat.h"
#include "pipeline/samplers/sampler_driver.h"
#include "pipeline/samplers/sampler_kernels.h"
#include "pipeline/samplers/torch_cuda_randn.h"
#include "runtime/device_arena.h"
#include "util/json.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
using namespace mi::pipeline;
using mi::Tensor;

fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "samplers"; }

const mi::json::Value& manifest() {
  static const mi::json::Value doc = [] {
    std::ifstream in(data_dir() / "manifest.json", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return mi::json::parse(ss.str());
  }();
  return doc;
}

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

std::vector<float> floats_from_bits(const mi::json::Value& arr) {
  std::vector<float> out;
  for (const auto& v : arr.as_array()) {
    const auto u = static_cast<std::uint32_t>(v.as_int());
    float f = 0;
    std::memcpy(&f, &u, 4);
    out.push_back(f);
  }
  return out;
}

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

int device_attr(cudaDeviceAttr a) {
  int dev = 0, v = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, dev));
  return v;
}

// Whether this GPU has the reference GPU's SM count and threads per SM (torch's randn grid then matches).
bool same_grid_as_reference() {
  return device_attr(cudaDevAttrMultiProcessorCount) == manifest().at("sm_count").as_int() &&
         device_attr(cudaDevAttrMaxThreadsPerMultiProcessor) == manifest().at("max_threads_per_sm").as_int();
}

struct DeviceBuffer {
  float* p = nullptr;
  std::int64_t n = 0;
  explicit DeviceBuffer(std::int64_t count) : n(count) { MI_CUDA_CHECK(cudaMalloc(&p, static_cast<std::size_t>(n) * 4)); }
  DeviceBuffer(const std::vector<float>& h) : DeviceBuffer(static_cast<std::int64_t>(h.size())) { upload(h); }
  ~DeviceBuffer() { cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void upload(const std::vector<float>& h) {
    MI_CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  }
  std::vector<float> download() const {
    std::vector<float> h(static_cast<std::size_t>(n));
    MI_CUDA_CHECK(cudaMemcpy(h.data(), p, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
  }
  Tensor tensor(const std::vector<std::int64_t>& shape) const {
    return Tensor::view(p, mi::DType::kF32, mi::Device::kCUDA, shape.data(), static_cast<int>(shape.size()));
  }
  Tensor flat() const { return Tensor::view(p, mi::DType::kF32, mi::Device::kCUDA, {n}); }
};

// Indices of the sampled elements the generator stored for large tensors.
std::vector<std::size_t> sample_indices(std::size_t n, std::size_t head, std::size_t stride, std::size_t tail) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < head; ++i) idx.push_back(i);
  for (std::size_t i = head; i < n; i += stride) idx.push_back(i);
  for (std::size_t i = n - tail; i < n; ++i) idx.push_back(i);
  return idx;
}

TEST(SamplerCudaRandn, MatchesTorchCudaGenerator) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const auto ref_sms = manifest().at("sm_count").as_int();
  const auto ref_threads = manifest().at("max_threads_per_sm").as_int();
  for (const auto& c : manifest().at("cuda_randn").as_array()) {
    const std::uint64_t seed = static_cast<std::uint64_t>(c.at("seed").as_int());
    const std::int64_t n = c.at("numel").as_int();
    // The reference GPU's grid, so the test does not depend on this GPU.
    const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((n + 255) / 256, ref_sms * (ref_threads / 256)));
    DeviceBuffer out(n);
    std::uint64_t offset = 0;
    const auto idx = sample_indices(static_cast<std::size_t>(n), 2048, static_cast<std::size_t>(c.at("sample_stride").as_int()), 1024);
    for (const auto& draw : c.at("draws").as_array()) {
      torch_cuda_randn(out.flat(), seed, offset, blocks, nullptr);
      offset += cuda_randn_offset_increment(n, blocks);
      const std::vector<float> got = out.download();
      const std::vector<float> want = floats_from_bits(draw);
      ASSERT_EQ(idx.size(), want.size());
      std::size_t bad = 0;
      for (std::size_t k = 0; k < idx.size(); ++k) bad += bits(got[idx[k]]) != bits(want[k]);
      EXPECT_EQ(bad, 0u) << "seed " << seed << " n " << n;
    }
  }
  // The generator object advances the offset the same way.
  if (same_grid_as_reference()) {
    const auto& c = manifest().at("cuda_randn").as_array()[0];
    const std::int64_t n = c.at("numel").as_int();
    TorchCudaGenerator gen(static_cast<std::uint64_t>(c.at("seed").as_int()), CudaRandnGrid::kDevice);
    DeviceBuffer out(n);
    gen.draw(out.flat(), nullptr);
    gen.draw(out.flat(), nullptr);
    EXPECT_EQ(bits(out.download()[0]), bits(floats_from_bits(c.at("draws").as_array()[1])[0]));
  }
  EXPECT_EQ(cuda_randn_blocks(1000, CudaRandnGrid::kUncapped), 4u);
  EXPECT_EQ(cuda_randn_offset_increment(1000, 4), 4u);
  EXPECT_EQ(cuda_randn_offset_increment(524288, 456), 8u);
}

TEST(SamplerCudaMath, VectorsComeFromThePinnedTorch) {
  // torch_sum200 follows torch >= 2.14's reduction order, and the expm1 comparison below counts the differences from
  // that build's CUDA expm1: the vectors must come from it (regenerate and re-check when the reference torch changes).
  EXPECT_EQ(manifest().at("torch").as_string(), "2.14.1+cu130");
}

TEST(SamplerCudaMath, Expm1CorrectlyRounded) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const std::vector<float> x = floats_from_bits(manifest().at("expm1").at("x"));
  const std::vector<float> torch_y = floats_from_bits(manifest().at("expm1").at("y"));
  DeviceBuffer dx(x), dy(static_cast<std::int64_t>(x.size()));
  sampler_kernels::expm1_rn(dx.p, dy.p, dx.n, nullptr);
  const std::vector<float> got = dy.download();
  // Host reference: glibc's double expm1 rounded once to fp32 (none of these inputs is a hard case for that rounding).
  // torch's CUDA expm1 is within 1 ulp of it everywhere, and differs on 234 of these 4028 inputs.
  std::size_t bad = 0, nans = 0, off_torch = 0;
  std::int64_t max_ulp_torch = 0;
  const auto ordered = [](float f) {
    const auto i = static_cast<std::int64_t>(static_cast<std::int32_t>(bits(f)));
    return i < 0 ? -(i & 0x7fffffff) : i;
  };
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float want = static_cast<float>(std::expm1(static_cast<double>(x[i])));
    if (std::isnan(want)) {
      bad += !std::isnan(got[i]);
      nans += std::isnan(torch_y[i]);
      continue;
    }
    bad += bits(got[i]) != bits(want);
    off_torch += bits(got[i]) != bits(torch_y[i]);
    max_ulp_torch = std::max(max_ulp_torch, std::abs(ordered(got[i]) - ordered(torch_y[i])));
  }
  EXPECT_EQ(bad, 0u) << "of " << got.size();
  EXPECT_GE(nans, 1u);
  EXPECT_EQ(off_torch, 234u);
  EXPECT_LE(max_ulp_torch, 1);
  std::printf("[samplers] expm1: %zu/%zu correctly rounded, %zu differ from torch by %lld ulp\n", got.size() - bad,
              got.size(), off_torch, static_cast<long long>(max_ulp_torch));
}

// The synthetic model of the generator: call k returns the CFG-combined denoised prediction of two velocity
// draws (torch CPU randn, seeds 1000 + 2k and 1001 + 2k), composed by guide_prediction() as denoise() composes the
// DiT rows, so the trajectories also cover denoise()'s model callback (heun's second call included).
struct SyntheticModel {
  std::int64_t n;
  double cfg;
  DeviceBuffer vc, vu;
  std::vector<float> sigmas_seen;
  SyntheticModel(std::int64_t numel, double g) : n(numel), cfg(g), vc(numel), vu(numel) {}
  void operator()(const Tensor& x, const ModelCall& call, const Tensor& out, cudaStream_t s) {
    sigmas_seen.push_back(call.sigma);
    EXPECT_EQ(call.timestep, call.sigma * 1000.0f);
    vc.upload(mi::rng::torch_cpu_randn(1000 + 2 * static_cast<std::uint64_t>(call.call), n));
    const Tensor v_cond = vc.tensor(std::vector<std::int64_t>(x.shape.begin(), x.shape.begin() + x.rank));
    const bool do_cfg = cfg != 1.0;
    Tensor v_uncond;
    if (do_cfg) {
      vu.upload(mi::rng::torch_cpu_randn(1001 + 2 * static_cast<std::uint64_t>(call.call), n));
      v_uncond = vu.tensor(std::vector<std::int64_t>(x.shape.begin(), x.shape.begin() + x.rank));
    }
    guide_prediction(Prediction::kDenoised, x, call.sigma, v_cond, v_uncond, Tensor{}, do_cfg, static_cast<float>(cfg),
                     false, 0.0f, out, s);
  }
};

// Samplers whose step coefficients use expm1 (the others reproduce the reference bit for bit).
bool uses_expm1(SamplerKind kind) {
  return kind == SamplerKind::kDpmpp2m || kind == SamplerKind::kDpmpp2mSde || kind == SamplerKind::kResMultistep;
}
constexpr double kExpm1TrajectoryTol = 1e-6;  // measured: at most 4.5e-7 (about 4 ulps of the largest element)

TEST(SamplerDriver, TrajectoriesMatchReference) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const bool same_grid = same_grid_as_reference();
  std::map<std::string, int> exact, total;
  std::map<std::string, double> worst_rel;
  int skipped = 0;
  for (const auto& e : manifest().at("trajectories").as_array()) {
    const std::string name = e.at("sampler").as_string();
    const SamplerKind kind = parse_sampler(name);
    std::vector<std::int64_t> shape;
    for (const auto& d : e.at("shape").as_array()) shape.push_back(d.as_int());
    shape.erase(shape.begin());  // batch 1
    std::int64_t n = 1;
    for (const auto d : shape) n *= d;
    const bool sampled = e.find("sample_stride") != nullptr;
    const bool capped = n > 256 * manifest().at("sm_count").as_int() * (manifest().at("max_threads_per_sm").as_int() / 256);
    if (capped && sampler_is_stochastic(kind) && !same_grid) {
      ++skipped;  // torch's CUDA noise for this size depends on the reference GPU's SM count
      continue;
    }
    const std::vector<float> sigmas = floats_from_bits(e.at("sigmas"));
    const auto seed = static_cast<std::uint64_t>(e.at("seed").as_int());
    SamplerOptions opts;
    opts.seed = seed;
    opts.noise = kind == SamplerKind::kDpmpp2mSde ? NoiseSource::kBrownianTree
                                                  : (capped ? NoiseSource::kPhiloxDevice : NoiseSource::kPhilox);
    opts.flow = FlowSampling{5.0, e.at("multiplier").as_double()};
    const std::string what = name + "/" + e.at("scheduler").as_string() + "/" + std::to_string(e.at("steps").as_int()) +
                             " " + e.at("file").as_string();

    SamplerDriver driver(kind, sigmas, shape, opts);
    EXPECT_EQ(driver.prediction(), Prediction::kDenoised);
    mi::runtime::DeviceArena arena(SamplerDriver::workspace_bytes(kind, n, driver.num_steps()));
    DeviceBuffer x(mi::rng::torch_cpu_randn(seed, n));
    flow_noise_scaling(x.tensor(shape), Tensor{}, sigmas[0], nullptr);
    SyntheticModel model(n, e.at("cfg").as_double());
    const int done = driver.run(
        x.tensor(shape), [&](const Tensor& in, const ModelCall& c, const Tensor& out, cudaStream_t s) { model(in, c, out, s); },
        arena, nullptr);
    EXPECT_EQ(done, driver.num_steps()) << what;
    EXPECT_EQ(static_cast<int>(model.sigmas_seen.size()), e.at("model_calls").as_int()) << what;
    EXPECT_EQ(driver.num_model_calls(), e.at("model_calls").as_int()) << what;
    const std::vector<float> want_sig = floats_from_bits(e.at("model_sigmas"));
    for (std::size_t k = 0; k < std::min(want_sig.size(), model.sigmas_seen.size()); ++k) {
      EXPECT_EQ(bits(model.sigmas_seen[k]), bits(want_sig[k])) << what << " call " << k;
    }

    const std::vector<float> got = x.download();
    const std::vector<float> want = mi::npy::load(data_dir() / e.at("file").as_string()).to_vector<float>();
    std::vector<std::size_t> idx;
    if (sampled) {
      idx = sample_indices(got.size(), 4096, static_cast<std::size_t>(e.at("sample_stride").as_int()), 2048);
    } else {
      for (std::size_t i = 0; i < got.size(); ++i) idx.push_back(i);
    }
    ASSERT_EQ(idx.size(), want.size()) << what;
    std::size_t bad = 0;
    double max_abs = 0, max_ref = 0;
    for (std::size_t k = 0; k < idx.size(); ++k) {
      bad += bits(got[idx[k]]) != bits(want[k]);
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[idx[k]]) - want[k]));
      max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[k])));
    }
    if (uses_expm1(kind)) {
      // The driver's expm1 is correctly rounded; torch's CUDA expm1 is 1 ulp off on 26.5M of the 2^32 fp32 inputs
      // (~0.6 %), and on 234 of the 4028 test vectors (~5.8 %). A coefficient that differs by 1 ulp moves the
      // trajectory by a few fp32 ulps of its largest element.
      EXPECT_LE(max_abs, kExpm1TrajectoryTol * max_ref) << what << ": " << bad << " elements differ";
      worst_rel[name] = std::max(worst_rel[name], max_abs / max_ref);
    } else {
      EXPECT_EQ(bad, 0u) << what << ": max |diff| " << max_abs;
    }
    ++total[name];
    exact[name] += bad == 0;
  }
  for (const auto& [name, t] : total) {
    std::printf("[samplers] %-16s %2d/%2d trajectories bit-exact", name.c_str(), exact[name], t);
    if (worst_rel.count(name) != 0) std::printf(", max |diff| / max |ref| %.3g", worst_rel[name]);
    std::printf("\n");
  }
  if (skipped > 0) std::printf("[samplers] %d large stochastic cases skipped: this GPU's SM count differs from the reference GPU\n", skipped);
}

TEST(SamplerDriver, VelocityEulerAndAb2MatchOfficialFormulas) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  // The official pipeline's step x += h * v (h rounded to fp32 from the exact difference) and AB2, against a host
  // evaluation of the same fp32 expressions.
  const std::vector<std::int64_t> shape = {128, 2, 3};
  const std::int64_t n = 128 * 6;
  const std::vector<float> sigmas = moss_schedule(6, 5.0).sigmas;
  for (const SamplerKind kind : {SamplerKind::kEuler, SamplerKind::kAb2}) {
    SamplerDriver driver(kind, sigmas, shape, SamplerOptions{}, /*official_euler=*/true);
    EXPECT_EQ(driver.prediction(), Prediction::kVelocity);
    mi::runtime::DeviceArena arena(SamplerDriver::workspace_bytes(kind, n, driver.num_steps()));
    std::vector<float> host = mi::rng::torch_cpu_randn(5, n);
    DeviceBuffer x(host);
    DeviceBuffer v(n);
    std::vector<float> v_prev;
    double h_prev = 0;
    int call = 0;
    driver.run(
        x.tensor(shape),
        [&](const Tensor&, const ModelCall& c, const Tensor& out, cudaStream_t) {
          const std::vector<float> vh = mi::rng::torch_cpu_randn(77 + static_cast<std::uint64_t>(call++), n);
          MI_CUDA_CHECK(cudaMemcpy(out.data, vh.data(), static_cast<std::size_t>(n) * 4, cudaMemcpyHostToDevice));
          const float s = c.sigma, sn = sigmas[static_cast<std::size_t>(c.step) + 1];
          const double h = static_cast<double>(sn) - static_cast<double>(s);
          for (std::int64_t i = 0; i < n; ++i) {
            const auto k = static_cast<std::size_t>(i);
            if (kind == SamplerKind::kEuler || c.step == 0) {
              const volatile float t = static_cast<float>(h) * vh[k];
              host[k] = host[k] + t;
            } else {
              const double r = h / (2.0 * h_prev);
              const volatile float a = static_cast<float>(1.0 + r) * vh[k];
              const volatile float b = static_cast<float>(r) * v_prev[k];
              const volatile float d = a - b;
              const volatile float t = static_cast<float>(h) * d;
              host[k] = host[k] + t;
            }
          }
          h_prev = h;
          v_prev = vh;
        },
        arena, nullptr);
    const std::vector<float> got = x.download();
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) bad += bits(got[i]) != bits(host[i]);
    EXPECT_EQ(bad, 0u) << sampler_name(kind);
  }
}

TEST(SamplerDriver, DataSpaceSampler) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  // x0 = 0; x = z + Q x0; v = model(x, code); x0 = z - v. Host reference in double for the channel mix.
  const int c = 8;
  const std::vector<std::int64_t> shape = {c, 3, 5};
  const std::int64_t pixels = 15, n = c * pixels;
  DataSpaceParams p;
  p.channels = c;
  p.q = mi::rng::torch_cpu_randn(9, c * c);
  for (float& v : p.q) v *= 0.05f;
  p.step_codes = {1000.0f, 968.0f, 926.0f, 871.0f};
  const std::vector<float> sigmas = dataspace_sigmas(p, 3);
  EXPECT_EQ(sigmas, (std::vector<float>{1.0f, 0.968f, 0.926f, 0.0f}));
  SamplerOptions opts;
  opts.dataspace = &p;
  SamplerDriver driver(SamplerKind::kDataSpace, sigmas, shape, opts);
  EXPECT_EQ(driver.prediction(), Prediction::kVelocity);
  mi::runtime::DeviceArena arena(SamplerDriver::workspace_bytes(SamplerKind::kDataSpace, n, 3, c));
  const std::vector<float> z = mi::rng::torch_cpu_randn(3, n);
  DeviceBuffer x(z);
  std::vector<float> x0(static_cast<std::size_t>(n), 0.0f);
  double max_err = 0;
  driver.run(
      x.tensor(shape),
      [&](const Tensor& in, const ModelCall& call, const Tensor& out, cudaStream_t) {
        EXPECT_EQ(call.timestep, p.step_codes[static_cast<std::size_t>(call.step)]);
        std::vector<float> xin(static_cast<std::size_t>(n));
        MI_CUDA_CHECK(cudaMemcpy(xin.data(), in.data, xin.size() * 4, cudaMemcpyDeviceToHost));
        for (int o = 0; o < c; ++o) {
          for (std::int64_t px = 0; px < pixels; ++px) {
            double acc = 0;
            for (int ch = 0; ch < c; ++ch) acc += static_cast<double>(p.q[static_cast<std::size_t>(o * c + ch)]) * x0[static_cast<std::size_t>(ch * pixels + px)];
            const double want = z[static_cast<std::size_t>(o * pixels + px)] + acc;
            max_err = std::max(max_err, std::fabs(want - xin[static_cast<std::size_t>(o * pixels + px)]));
          }
        }
        const std::vector<float> v = mi::rng::torch_cpu_randn(50 + static_cast<std::uint64_t>(call.step), n);
        MI_CUDA_CHECK(cudaMemcpy(out.data, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        for (std::size_t i = 0; i < x0.size(); ++i) x0[i] = z[i] - v[i];
      },
      arena, nullptr);
  EXPECT_LT(max_err, 1e-6);
  const std::vector<float> got = x.download();
  std::size_t bad = 0;
  for (std::size_t i = 0; i < got.size(); ++i) bad += bits(got[i]) != bits(x0[i]);
  EXPECT_EQ(bad, 0u);
  std::printf("[samplers] dataspace channel mix max |err| vs double %.3g\n", max_err);
}

TEST(SamplerDriver, DataSpaceMixMatchesTorchEinsum) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const fs::path ds = "/workspace/Models/community/lodestones/MOSS-Image-SFT-8-steps-dmd/dataspace.safetensors";
  if (!fs::is_regular_file(ds) || !fs::is_regular_file(data_dir() / "dataspace_mix.npy")) GTEST_SKIP() << ds << " absent";
  // z + einsum("oc,bchw->bohw", Q, x0) as the reference runs it (cuBLAS fp32): not bit-exact (summation order), so the
  // error is measured against the reference's magnitude.
  const DataSpaceParams p = load_dataspace(ds.string());
  const std::int64_t pixels = 16 * 16, n = 128 * pixels;
  DeviceBuffer x0(mi::rng::torch_cpu_randn(11, n)), z(mi::rng::torch_cpu_randn(12, n)), q(p.q), out(n);
  sampler_kernels::dataspace_mix(z.p, x0.p, q.p, 128, pixels, out.p, nullptr);
  const std::vector<float> got = out.download();
  const std::vector<float> want = mi::npy::load(data_dir() / "dataspace_mix.npy").to_vector<float>();
  ASSERT_EQ(got.size(), want.size());
  double max_abs = 0, dot = 0, ng = 0, nw = 0;
  std::size_t exact = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
    dot += static_cast<double>(got[i]) * want[i];
    ng += static_cast<double>(got[i]) * got[i];
    nw += static_cast<double>(want[i]) * want[i];
    exact += bits(got[i]) == bits(want[i]);
  }
  const double cosine = dot / std::sqrt(ng * nw);
  std::printf("[samplers] dataspace mix vs torch einsum: max |diff| %.3g, cosine %.9f, %zu/%zu bit-exact\n", max_abs,
              cosine, exact, got.size());
  EXPECT_LT(max_abs, 1e-6);
  EXPECT_GT(cosine, 0.9999999);
}

TEST(SamplerDriver, ArgumentChecksAndNames) {
  EXPECT_THROW(SamplerDriver(SamplerKind::kEuler, {1.0f}, {4}, {}), std::invalid_argument);
  EXPECT_THROW(SamplerDriver(SamplerKind::kEuler, {0.5f, 0.7f, 0.0f}, {4}, {}), std::invalid_argument);
  EXPECT_THROW(SamplerDriver(SamplerKind::kDataSpace, {1.0f, 0.0f}, {4}, {}), std::invalid_argument);
  SamplerOptions bad;
  bad.eta = -1;
  EXPECT_THROW(SamplerDriver(SamplerKind::kEulerAncestral, {1.0f, 0.0f}, {4}, bad), std::invalid_argument);
  for (const char* s : {"euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep", "dataspace"}) {
    EXPECT_STREQ(sampler_name(parse_sampler(s)), s);
  }
  EXPECT_THROW(parse_sampler("lms"), std::invalid_argument);
  for (const char* s : {"philox", "philox_device", "brownian_tree"}) EXPECT_STREQ(noise_source_name(parse_noise_source(s)), s);
  // The SDE samplers move a first sigma of 1 off the log-SNR singularity.
  SamplerOptions o;
  const SamplerDriver sde(SamplerKind::kDpmpp2mSde, {1.0f, 0.5f, 0.0f}, {4}, o);
  EXPECT_EQ(sde.sigmas()[0], static_cast<float>(flow_percent_to_sigma(o.flow, 1e-4)));
  const SamplerDriver heun(SamplerKind::kHeun, {1.0f, 0.5f, 0.0f}, {4}, o);
  EXPECT_EQ(heun.num_model_calls(), 3);
}

TEST(SamplerDriver, LoadDataspaceFiles) {
  int found = 0;
  for (const char* repo : {"lodestones/MOSS-Image-SFT-8-steps-dmd", "darask0/MOSS-Image_4-step_adapter_lora"}) {
    const fs::path p = fs::path("/workspace/Models/community") / repo / "dataspace.safetensors";
    if (!fs::is_regular_file(p)) continue;
    ++found;
    const DataSpaceParams d = load_dataspace(p.string());
    EXPECT_EQ(d.channels, 128);
    EXPECT_EQ(d.q.size(), 128u * 128u);
    ASSERT_GE(d.step_codes.size(), 4u);
    EXPECT_EQ(d.step_codes[0], 1000.0f);
    EXPECT_EQ(d.step_codes[1], 968.0f);
    EXPECT_EQ(dataspace_sigmas(d, 4).back(), 0.0f);
    EXPECT_THROW(dataspace_sigmas(d, static_cast<int>(d.step_codes.size()) + 1), std::invalid_argument);
  }
  if (found == 0) GTEST_SKIP() << "no dataspace.safetensors under /workspace/Models/community";
}

TEST(SamplerDriver, MalformedDataspaceRaisesDataspaceError) {
  const fs::path dir = fs::temp_directory_path() / "mossinfer_dataspace_test";
  fs::create_directories(dir);
  const auto write = [&](const char* name, const std::string& bytes) {
    const fs::path p = dir / name;
    std::ofstream(p, std::ios::binary) << bytes;
    return p.string();
  };
  // Not a safetensors file at all.
  EXPECT_THROW(load_dataspace(write("garbage.safetensors", "this is not a safetensors file")), DataspaceError);
  // A valid, empty safetensors file: the expected tensors are missing.
  const std::string header = "{}";
  std::string empty(8, '\0');
  empty[0] = static_cast<char>(header.size());
  try {
    load_dataspace(write("empty.safetensors", empty + header));
    ADD_FAILURE() << "expected DataspaceError";
  } catch (const DataspaceError& e) {
    EXPECT_NE(std::string(e.what()).find("expected tensors latent_mix.Q and step_codes"), std::string::npos) << e.what();
  }
  fs::remove_all(dir);
}

TEST(SamplerDriver, CancelViaAfterStep) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const std::vector<std::int64_t> shape = {128, 2, 2};
  const std::int64_t n = 512;
  for (const SamplerKind kind : {SamplerKind::kEuler, SamplerKind::kHeun, SamplerKind::kEulerAncestral}) {
    SamplerOptions o;
    o.seed = 1;
    SamplerDriver driver(kind, compute_sigmas(SchedulerKind::kSimple, 10, FlowSampling{}), shape, o);
    mi::runtime::DeviceArena arena(SamplerDriver::workspace_bytes(kind, n, driver.num_steps()));
    DeviceBuffer x(mi::rng::torch_cpu_randn(2, n));
    int calls = 0, last_step = -1;
    std::vector<int> finished;
    const int run = driver.run(
        x.tensor(shape),
        [&](const Tensor& in, const ModelCall& c, const Tensor& out, cudaStream_t s) {
          ++calls;
          last_step = c.step;
          velocity_to_denoised(in, in, c.sigma, out, s);
        },
        arena, nullptr, [&](int f) {
          finished.push_back(f);
          return f < 3;
        });
    EXPECT_EQ(run, 3) << sampler_name(kind);
    EXPECT_EQ(finished, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(last_step, 2);
    EXPECT_EQ(calls, 3 * model_calls_per_step(kind));
    EXPECT_EQ(arena.stack_used(), 0u);  // scratch released on the early return
    EXPECT_THROW(driver.run(x.tensor(shape), [](const Tensor&, const ModelCall&, const Tensor&, cudaStream_t) {}, arena,
                            nullptr),
                 std::logic_error);  // a driver runs once
  }
}

// Host fp32 reference of guide_prediction(), one rounding per op (volatile keeps the compiler from contracting).
float hsub(float a, float b) { volatile float r = a - b; return r; }
float hadd(float a, float b) { volatile float r = a + b; return r; }
float hmul(float a, float b) { volatile float r = a * b; return r; }

TEST(SamplerDriver, GuidePredictionMatchesReferenceFormulas) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  const std::int64_t n = 4096;
  const std::vector<float> x = mi::rng::torch_cpu_randn(21, n), vc = mi::rng::torch_cpu_randn(22, n),
                           vu = mi::rng::torch_cpu_randn(23, n), vk = mi::rng::torch_cpu_randn(24, n);
  DeviceBuffer dx(x), dc(vc), du(vu), dk(vk), dout(n);
  const float sigma = 0.73f, g = 3.5f, sc = 2.25f;
  struct Case {
    Prediction space;
    bool cfg, slg;
  };
  for (const Case c : {Case{Prediction::kVelocity, true, true}, Case{Prediction::kVelocity, true, false},
                       Case{Prediction::kVelocity, false, true}, Case{Prediction::kVelocity, false, false},
                       Case{Prediction::kDenoised, true, true}, Case{Prediction::kDenoised, false, true},
                       Case{Prediction::kDenoised, false, false}}) {
    guide_prediction(c.space, dx.flat(), sigma, dc.flat(), c.cfg ? du.flat() : Tensor{}, c.slg ? dk.flat() : Tensor{},
                     c.cfg, g, c.slg, sc, dout.flat(), nullptr);
    const std::vector<float> got = dout.download();
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      float want = 0;
      if (c.space == Prediction::kVelocity) {
        // The official pipeline: uncond + g * (cond - uncond), then pred + s * (cond - skipped), pred = skipped at cfg 1.
        want = vc[i];
        if (c.cfg) want = hadd(vu[i], hmul(g, hsub(vc[i], vu[i])));
        if (c.slg) want = hadd(c.cfg ? want : vk[i], hmul(sc, hsub(vc[i], vk[i])));
      } else {
        // k-diffusion on denoised predictions: D = x - v * sigma; u + (c - u) * g; D + (c - k) * s.
        const float d_c = hsub(x[i], hmul(vc[i], sigma)), d_u = hsub(x[i], hmul(vu[i], sigma));
        const float d_k = hsub(x[i], hmul(vk[i], sigma));
        want = c.cfg ? hadd(d_u, hmul(hsub(d_c, d_u), g)) : d_c;
        if (c.slg) want = hadd(want, hmul(hsub(d_c, d_k), sc));
      }
      bad += bits(got[i]) != bits(want);
    }
    EXPECT_EQ(bad, 0u) << (c.space == Prediction::kVelocity ? "velocity" : "denoised") << " cfg " << c.cfg << " slg "
                       << c.slg;
  }
}

TEST(SamplerDriver, RepeatedSigmasAndOptionQueries) {
  // Zero-length steps: rejected where step-size ratios divide by zero, accepted by the euler-type samplers.
  const std::vector<float> repeated = {1.0f, 0.5f, 0.5f, 0.2f, 0.0f};
  for (const SamplerKind k : {SamplerKind::kAb2, SamplerKind::kDpmpp2m, SamplerKind::kDpmpp2mSde, SamplerKind::kErSde,
                              SamplerKind::kResMultistep}) {
    EXPECT_THROW(SamplerDriver(k, repeated, {4}, {}), std::invalid_argument) << sampler_name(k);
  }
  for (const SamplerKind k : {SamplerKind::kEuler, SamplerKind::kEulerAncestral, SamplerKind::kHeun}) {
    EXPECT_NO_THROW(SamplerDriver(k, repeated, {4}, {})) << sampler_name(k);
  }
  // A simple schedule longer than the 1000-entry table repeats sigmas.
  EXPECT_THROW(SamplerDriver(SamplerKind::kDpmpp2m, compute_sigmas(SchedulerKind::kSimple, 1500, FlowSampling{}), {4}, {}),
               std::invalid_argument);
  EXPECT_EQ(SamplerDriver(SamplerKind::kDpmpp2m, {1.0f, 0.0f, 0.0f}, {4}, {}).num_steps(), 2);  // repeated 0 is fine

  for (const SamplerKind k : {SamplerKind::kEuler, SamplerKind::kAb2, SamplerKind::kHeun, SamplerKind::kDpmpp2m,
                              SamplerKind::kResMultistep}) {
    EXPECT_FALSE(sampler_needs_options(k));
    EXPECT_TRUE(sampler_supported_by(k, false));
  }
  for (const SamplerKind k : {SamplerKind::kEulerAncestral, SamplerKind::kDpmpp2mSde, SamplerKind::kErSde,
                              SamplerKind::kDataSpace}) {
    EXPECT_TRUE(sampler_needs_options(k));
    EXPECT_FALSE(sampler_supported_by(k, false));
    EXPECT_TRUE(sampler_supported_by(k, true));
  }
  EXPECT_EQ(model_calls_per_step(SamplerKind::kHeun), 2);
  EXPECT_EQ(model_calls_per_step(SamplerKind::kEuler), 1);
  EXPECT_EQ(model_calls_per_step(SamplerKind::kDpmpp2mSde), 1);
  // official_euler only changes euler.
  EXPECT_EQ(SamplerDriver(SamplerKind::kEuler, {1.0f, 0.0f}, {4}, {}, true).prediction(), Prediction::kVelocity);
  EXPECT_EQ(SamplerDriver(SamplerKind::kEuler, {1.0f, 0.0f}, {4}, {}).prediction(), Prediction::kDenoised);
  EXPECT_EQ(SamplerDriver(SamplerKind::kHeun, {1.0f, 0.0f}, {4}, {}, true).prediction(), Prediction::kDenoised);
}

}  // namespace
