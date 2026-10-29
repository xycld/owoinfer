// P6b accelerators (pipeline/accel): option parsing and validation, CFG truncation, the FBCache kernels and decisions
// against host arithmetic, and on the real DiT: every accelerator off (or a cache that never reuses) is bit-identical
// to the plain loop, resident, streamed and with CUDA graphs alike; a reuse skips the blocks (and the streaming).
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/accel/accel.h"
#include "pipeline/accel/accel_kernels.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "pipeline/sampler.h"
#include "pipeline/samplers/schedules.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "weights/model_index.h"
#include "test_util.h"

namespace {

namespace fs = std::filesystem;
namespace accel = mi::pipeline::accel;
using mi::Tensor;
using mi::moss_dit::MossDit;
using mi::moss_dit::Row;
using mi::moss_dit::SkipSet;

std::uint16_t to_bf16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // finite inputs only
}
float from_bf16(std::uint16_t h) {
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float v = 0;
  std::memcpy(&v, &u, 4);
  return v;
}
std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<std::uint16_t> v(n);
  for (auto& x : v) x = to_bf16(d(rng));
  return v;
}

// ---- CPU ----

TEST(Accel, OptionsParseValidateDescribe) {
  EXPECT_EQ(accel::parse_step_cache("off"), accel::StepCacheKind::kOff);
  EXPECT_EQ(accel::parse_step_cache("fbcache"), accel::StepCacheKind::kFbCache);
  EXPECT_STREQ(accel::step_cache_name(accel::StepCacheKind::kFbCache), "fbcache");
  try {
    accel::parse_step_cache("teacache");
    FAIL() << "teacache accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("fbcache"), std::string::npos);
  }
  EXPECT_THROW(accel::parse_step_cache("FBCache"), std::invalid_argument);

  accel::Options o;  // defaults: everything off
  EXPECT_FALSE(o.active(true));
  EXPECT_FALSE(o.active(false));
  EXPECT_EQ(o.describe(true), "off");
  EXPECT_NO_THROW(accel::validate(o));
  o.cfg_truncation = 0.8;
  EXPECT_TRUE(o.active(true));
  EXPECT_FALSE(o.active(false));  // no CFG: nothing to truncate
  EXPECT_EQ(o.describe(true), "cfg_truncation 0.8");
  o.step_cache = accel::StepCacheKind::kFbCache;
  o.step_cache_threshold = 0.1;
  EXPECT_TRUE(o.active(false));
  EXPECT_EQ(o.describe(true), "cfg_truncation 0.8, step_cache fbcache 0.1");
  for (const double bad : {-0.1, 1.5, std::nan("")}) {
    accel::Options b;
    b.cfg_truncation = bad;
    EXPECT_THROW(accel::validate(b), std::invalid_argument) << bad;
  }
  for (const double bad : {0.0, -1.0, 1.01, std::numeric_limits<double>::infinity()}) {
    accel::Options b;
    b.step_cache_threshold = bad;
    EXPECT_NO_THROW(accel::validate(b)) << bad << ": the threshold is not used (nor checked) without the cache";
    b.step_cache = accel::StepCacheKind::kFbCache;
    EXPECT_THROW(accel::validate(b), std::invalid_argument) << bad;
  }
  accel::Options edge;
  edge.cfg_truncation = 0.0;
  edge.step_cache_threshold = 1.0;
  EXPECT_NO_THROW(accel::validate(edge));
  // An unset threshold is the model's default: 0.02 on the base model, 0.03 otherwise (the pipeline fills it in).
  accel::Options dflt;
  dflt.step_cache = accel::StepCacheKind::kFbCache;
  EXPECT_NO_THROW(accel::validate(dflt));
  EXPECT_EQ(dflt.describe(false), "step_cache fbcache default");
  EXPECT_EQ(accel::default_fbcache_threshold("base"), accel::kBaseModelFbCacheThreshold);
  EXPECT_EQ(accel::default_fbcache_threshold("sft"), accel::kDefaultFbCacheThreshold);
  EXPECT_EQ(accel::default_fbcache_threshold(""), accel::kDefaultFbCacheThreshold);
}

TEST(Accel, CfgTruncationWindow) {
  mi::pipeline::Guidance g;  // cfg 3, no truncation
  for (int i = 0; i < 50; ++i) EXPECT_TRUE(g.cfg_active(i, 50));
  g.cfg_truncation = 0.8;
  int active = 0;
  for (int i = 0; i < 50; ++i) active += g.cfg_active(i, 50);
  EXPECT_EQ(active, 40);  // steps 40..49 (the last 20 %) run the cond row alone
  EXPECT_TRUE(g.cfg_active(39, 50));
  EXPECT_FALSE(g.cfg_active(40, 50));
  g.cfg_truncation = 0.0;
  for (int i = 0; i < 10; ++i) EXPECT_FALSE(g.cfg_active(i, 10));
  g.cfg_truncation = 0.5;
  g.cfg = 1.0;  // no CFG at all
  for (int i = 0; i < 10; ++i) EXPECT_FALSE(g.cfg_active(i, 10));
}

// 1.1: the threshold is optional (unset = the model's default). parameters.json: absent or null leaves it unset; the
// report shows the request's (null when unset) and what ran with its source; the PNG metadata records what ran; the
// warnings: a set threshold above 0.02 on the base model, and a cache that never engaged.
TEST(Accel, OptionalThresholdParamsReportMetadataAndWarnings) {
  using mi::pipeline::request_from_parameters;
  EXPECT_FALSE(request_from_parameters(mi::json::parse(R"({"prompt": "a", "step_cache": "fbcache"})")).accel.step_cache_threshold);
  EXPECT_FALSE(request_from_parameters(mi::json::parse(R"({"prompt": "a", "step_cache": "fbcache", "step_cache_threshold": null})"))
                   .accel.step_cache_threshold);
  const mi::pipeline::Request q = request_from_parameters(mi::json::parse(R"({"prompt": "a", "step_cache": "fbcache"})"));
  EXPECT_NO_THROW(mi::pipeline::validate(q));
  mi::pipeline::Result r;
  r.accel.dit_calls = 10;
  r.accel.step_cache_threshold = accel::kBaseModelFbCacheThreshold;  // what the pipeline fills in on the base model
  r.accel.step_cache_threshold_source = accel::threshold_source(q.accel, "base");
  EXPECT_EQ(r.accel.step_cache_threshold_source, "default for the base model");
  EXPECT_EQ(accel::threshold_source(q.accel, ""), "default for the unrecognized model");
  const mi::json::Value rep = mi::json::parse("{" + mi::pipeline::report_fields(q, r, "", "") + "}");
  EXPECT_TRUE(rep.at("request").at("step_cache_threshold").is_null());
  EXPECT_EQ(rep.at("accelerators").at("step_cache_threshold").as_double(), 0.02);
  EXPECT_EQ(rep.at("accelerators").at("step_cache_threshold_source").as_string(), "default for the base model");
  EXPECT_EQ(rep.at("accelerators").at("active").as_string(), "step_cache fbcache 0.02");
  bool found = false;
  for (const auto& [k, v] : mi::pipeline::generation_metadata(q, r).extra) {
    if (k == "step_cache_threshold") {
      EXPECT_EQ(v, "0.02");
      found = true;
    }
  }
  EXPECT_TRUE(found) << "the metadata records the threshold that ran";

  accel::Options set;
  set.step_cache = accel::StepCacheKind::kFbCache;
  set.step_cache_threshold = 0.03;
  EXPECT_EQ(accel::threshold_source(set, "base"), "request");
  EXPECT_NE(accel::base_model_threshold_warning(set, "base").find("fbcache 0.03 on the base model"), std::string::npos);
  EXPECT_EQ(accel::base_model_threshold_warning(set, "sft"), "");
  EXPECT_EQ(accel::base_model_threshold_warning(q.accel, "base"), "") << "the default is never warned about";
  set.step_cache_threshold = 0.02;
  EXPECT_EQ(accel::base_model_threshold_warning(set, "base"), "");

  accel::Stats st;
  st.step_cache_threshold = 0.05;
  st.dit_calls = 6;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  st.step_cache_diffs = {nan, 0.09, 0.07, nan, 0.11};  // three comparable calls: not enough evidence
  EXPECT_EQ(accel::never_engaged_warning(st), "");
  st.step_cache_diffs.push_back(0.08);
  const std::string w = accel::never_engaged_warning(st);
  EXPECT_NE(w.find("never engaged"), std::string::npos) << w;
  EXPECT_NE(w.find("at least 0.07"), std::string::npos) << w;
  EXPECT_NE(w.find("over 4 comparable"), std::string::npos) << w;
  st.cached_calls = 1;
  EXPECT_EQ(accel::never_engaged_warning(st), "");
}

TEST(Accel, RequestParametersValidationAndReport) {
  const mi::json::Value doc = mi::json::parse(
      R"({"prompt": "a cat", "cfg_truncation": 0.75, "step_cache": "fbcache", "step_cache_threshold": 0.04})");
  mi::pipeline::Request q = mi::pipeline::request_from_parameters(doc);
  EXPECT_EQ(q.accel.cfg_truncation, 0.75);
  EXPECT_EQ(q.accel.step_cache, accel::StepCacheKind::kFbCache);
  EXPECT_EQ(q.accel.step_cache_threshold, 0.04);
  EXPECT_NO_THROW(mi::pipeline::validate(q));
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"prompt": "a", "step_cache": "teacache"})")),
               std::invalid_argument);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"prompt": "a", "step_cache": 1})")),
               std::invalid_argument);
  mi::pipeline::Request bad = q;
  bad.accel.cfg_truncation = 2;
  EXPECT_THROW(mi::pipeline::validate(bad), std::invalid_argument);
  // Defaults: off.
  const mi::pipeline::Request plain = mi::pipeline::request_from_parameters(mi::json::parse(R"({"prompt": "a"})"));
  EXPECT_FALSE(plain.accel.active(true));

  mi::pipeline::Result r;
  r.accel.cfg_truncated_steps = 5;
  r.accel.dit_calls = 20;
  r.accel.cached_calls = 7;
  r.accel.cached_steps = {3, 5, 6};
  const mi::json::Value rep = mi::json::parse("{" + mi::pipeline::report_fields(q, r, "o.png", "") + "}");
  const mi::json::Value& a = rep.at("accelerators");
  EXPECT_TRUE(a.at("lossy").as_bool());
  EXPECT_EQ(a.at("active").as_string(), "cfg_truncation 0.75, step_cache fbcache 0.04");
  EXPECT_EQ(a.at("cfg_truncated_steps").as_int(), 5);
  EXPECT_EQ(a.at("step_cache_calls").as_int(), 20);
  EXPECT_EQ(a.at("step_cache_reused").as_int(), 7);
  EXPECT_EQ(a.at("cached_steps").as_array().size(), 3u);
  EXPECT_EQ(rep.at("request").at("step_cache").as_string(), "fbcache");
  EXPECT_EQ(rep.at("request").at("cfg_truncation").as_double(), 0.75);
  const mi::json::Value off = mi::json::parse("{" + mi::pipeline::report_fields(plain, mi::pipeline::Result{}, "", "") + "}");
  EXPECT_FALSE(off.at("accelerators").at("lossy").as_bool());
  EXPECT_EQ(off.at("accelerators").at("active").as_string(), "off");

  // PNG metadata: only the accelerators that changed the image.
  const auto has = [](const mi::pipeline::GenerationMetadata& m, const std::string& key) {
    for (const auto& [k, v] : m.extra) {
      if (k == key) return true;
    }
    return false;
  };
  const mi::pipeline::GenerationMetadata m = mi::pipeline::generation_metadata(q, r);
  EXPECT_TRUE(has(m, "cfg_truncation"));
  EXPECT_TRUE(has(m, "step_cache"));
  EXPECT_TRUE(has(m, "step_cache_threshold"));
  const mi::pipeline::GenerationMetadata m0 = mi::pipeline::generation_metadata(plain, r);
  EXPECT_FALSE(has(m0, "cfg_truncation"));
  EXPECT_FALSE(has(m0, "step_cache"));
}

TEST(Accel, CacheWorkspaceBytes) {
  const std::size_t buf = mi::runtime::align_up(std::size_t{64} * 64 * mi::moss_dit::kDim * 2);
  const std::size_t partials = mi::runtime::align_up(std::size_t{2} * mi::pipeline::accel_kernels::kDiffBlocks * 2 * 8);
  EXPECT_EQ(accel::FirstBlockCache::workspace_bytes(64, 64, 2), 6 * buf + partials);
  EXPECT_EQ(accel::FirstBlockCache::workspace_bytes(64, 64, 2), (std::size_t{144} << 20) + partials);  // 72 MiB a row
}

// ---- GPU: kernels and cache decisions on synthetic residual streams ----

std::vector<std::uint16_t> download_bf16(const void* p, std::size_t n) {
  std::vector<std::uint16_t> h(n);
  MI_CUDA_CHECK(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
  return h;
}

TEST(GpuAccel, KernelsMatchHostArithmetic) {
  const std::int64_t n = 3 * 3072 + 17;
  mi::runtime::DeviceArena arena(std::size_t{8} << 20);
  const Tensor a = arena.alloc_tensor(mi::DType::kBF16, {n}), b = arena.alloc_tensor(mi::DType::kBF16, {n});
  const Tensor c = arena.alloc_tensor(mi::DType::kBF16, {n}), out = arena.alloc_tensor(mi::DType::kBF16, {n});
  auto* partials = static_cast<double*>(arena.alloc_stack(mi::pipeline::accel_kernels::kDiffBlocks * 2 * sizeof(double)));
  const auto ha = random_bf16(static_cast<std::size_t>(n), 1, 3.0f), hb = random_bf16(static_cast<std::size_t>(n), 2, 3.0f);
  const auto hc = random_bf16(static_cast<std::size_t>(n), 3, 0.5f);
  MI_CUDA_CHECK(cudaMemcpy(a.data, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(b.data, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(c.data, hc.data(), hc.size() * 2, cudaMemcpyHostToDevice));
  // A pageable H2D cudaMemcpy may return before its DMA lands; wait for it, as DeviceBuffer::upload does.
  MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));

  mi::pipeline::accel_kernels::sub_bf16(a.data, b.data, out.data, n, nullptr);
  auto got = download_bf16(out.data, static_cast<std::size_t>(n));
  double num = 0, den = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const std::uint16_t want = to_bf16(from_bf16(ha[i]) - from_bf16(hb[i]));
    ASSERT_EQ(got[i], want) << i;
    num += std::fabs(static_cast<double>(from_bf16(want)) - from_bf16(hc[i]));
    den += std::fabs(static_cast<double>(from_bf16(hc[i])));
  }
  mi::pipeline::accel_kernels::residual_diff_partials(a.data, b.data, c.data, n, partials, nullptr);
  std::vector<double> hp(mi::pipeline::accel_kernels::kDiffBlocks * 2);
  MI_CUDA_CHECK(cudaMemcpy(hp.data(), partials, hp.size() * 8, cudaMemcpyDeviceToHost));
  double gn = 0, gd = 0;
  for (std::size_t i = 0; i < hp.size(); i += 2) gn += hp[i], gd += hp[i + 1];
  EXPECT_NEAR(gn, num, 1e-9 * num);
  EXPECT_NEAR(gd, den, 1e-9 * den);
  // Deterministic: the same partial sums bit for bit on a second run.
  mi::pipeline::accel_kernels::residual_diff_partials(a.data, b.data, c.data, n, partials, nullptr);
  std::vector<double> hp2(hp.size());
  MI_CUDA_CHECK(cudaMemcpy(hp2.data(), partials, hp2.size() * 8, cudaMemcpyDeviceToHost));
  EXPECT_EQ(std::memcmp(hp.data(), hp2.data(), hp.size() * 8), 0);

  mi::pipeline::accel_kernels::add_bf16(a.data, b.data, n, nullptr);
  got = download_bf16(a.data, static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], to_bf16(from_bf16(ha[i]) + from_bf16(hb[i]))) << i;
}

// A fake forward around the hook: x0 -> before_blocks; x = x0 + r0 (block 0) -> after_first_block; unless reused,
// x = x1 + R (the other blocks) -> after_blocks. Returns the rows' final video tokens.
struct FakeDit {
  mi::runtime::DeviceArena& arena;
  std::int64_t video;
  std::vector<std::uint16_t> run(accel::FirstBlockCache& cache, const std::vector<Row>& rows, const std::vector<float>& r0,
                                 const std::vector<float>& rest, bool* reused) {
    const std::int64_t s = 4;  // text tokens per row
    std::vector<std::int64_t> off{0};
    for (std::size_t r = 0; r < rows.size(); ++r) off.push_back(off.back() + video + s);
    const auto n = static_cast<std::size_t>(video * mi::moss_dit::kDim);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(off.back()) * mi::moss_dit::kDim, to_bf16(0.25f));
    mi::runtime::StackScope scope(arena);
    const Tensor dx = arena.alloc_tensor(mi::DType::kBF16, {off.back(), mi::moss_dit::kDim});
    // A pageable H2D cudaMemcpy may return before its DMA lands; wait for it, as DeviceBuffer::upload does.
    const auto upload = [&] {
      MI_CUDA_CHECK(cudaMemcpy(dx.data, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    };
    const auto download = [&] { MI_CUDA_CHECK(cudaMemcpy(x.data(), dx.data, x.size() * 2, cudaMemcpyDeviceToHost)); };
    const auto video_at = [&](std::size_t r, std::size_t i) -> std::uint16_t& {
      return x[static_cast<std::size_t>(off[r]) * mi::moss_dit::kDim + i];
    };
    for (std::size_t r = 0; r < rows.size(); ++r) {
      for (std::size_t i = 0; i < n; ++i) video_at(r, i) = to_bf16(0.5f + 0.001f * static_cast<float>(i % 97));
    }
    upload();
    cache.before_blocks(dx, rows, off, video, nullptr);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      for (std::size_t i = 0; i < n; ++i) video_at(r, i) = to_bf16(from_bf16(video_at(r, i)) + r0[i % r0.size()] * (1.0f + static_cast<float>(r)));
    }
    upload();
    *reused = cache.after_first_block(dx, rows, off, video, nullptr);
    if (!*reused) {
      download();
      for (std::size_t r = 0; r < rows.size(); ++r) {
        for (std::size_t i = 0; i < n; ++i) video_at(r, i) = to_bf16(from_bf16(video_at(r, i)) + rest[i % rest.size()] - static_cast<float>(r));
      }
      upload();
    }
    cache.after_blocks(dx, rows, off, video, *reused, nullptr);
    download();
    std::vector<std::uint16_t> out;
    for (std::size_t r = 0; r < rows.size(); ++r) out.insert(out.end(), &video_at(r, 0), &video_at(r, 0) + n);
    return out;
  }
};

TEST(GpuAccel, FirstBlockCacheDecisionsAndReuse) {
  const std::int64_t video = 8;
  mi::runtime::DeviceArena arena(std::size_t{64} << 20);
  mi::runtime::StackScope scope(arena);
  const Tensor cond = arena.alloc_tensor(mi::DType::kBF16, {4, mi::moss_dit::kDim});
  const Tensor uncond = arena.alloc_tensor(mi::DType::kBF16, {4, mi::moss_dit::kDim});
  SkipSet skip22;
  skip22.set(22);
  const std::vector<Row> both = {{cond, {}}, {uncond, skip22}};
  accel::FirstBlockCache cache(arena, 2, 4, 3, 0.1);  // 8 video tokens
  FakeDit dit{arena, video};
  const std::vector<float> r0 = {0.5f, -0.25f, 1.0f};
  const std::vector<float> rest = {2.0f, 0.75f, -1.5f, 0.125f};
  bool reused = true;
  const auto full = dit.run(cache, both, r0, rest, &reused);
  EXPECT_FALSE(reused);  // nothing to compare with
  EXPECT_TRUE(std::isnan(cache.diffs().back()));

  // Same block-0 residual: diff 0, reuse; the rows' tokens are x1 + R of the full call, rounded once each.
  const auto again = dit.run(cache, both, r0, rest, &reused);
  EXPECT_TRUE(reused);
  EXPECT_EQ(cache.diffs().back(), 0.0);
  ASSERT_EQ(again.size(), full.size());
  const auto n = static_cast<std::size_t>(video * mi::moss_dit::kDim);
  for (std::size_t r = 0; r < 2; ++r) {
    for (std::size_t i = 0; i < n; ++i) {
      const float x0 = from_bf16(to_bf16(0.5f + 0.001f * static_cast<float>(i % 97)));
      const float x1 = from_bf16(to_bf16(x0 + r0[i % r0.size()] * (1.0f + static_cast<float>(r))));
      const float xf = from_bf16(full[r * n + i]);
      const float big_r = from_bf16(to_bf16(xf - x1));
      ASSERT_EQ(again[r * n + i], to_bf16(x1 + big_r)) << r << " " << i;
    }
  }

  // A 6 % larger residual: relative change ~0.06 < 0.1 reuses, 20 % does not (and refreshes the reference).
  std::vector<float> r0b = r0;
  for (float& v : r0b) v *= 1.06f;
  dit.run(cache, both, r0b, rest, &reused);
  EXPECT_TRUE(reused);
  EXPECT_NEAR(cache.diffs().back(), 0.06, 0.01);
  std::vector<float> r0c = r0;
  for (float& v : r0c) v *= 1.2f;
  dit.run(cache, both, r0c, rest, &reused);
  EXPECT_FALSE(reused);
  EXPECT_NEAR(cache.diffs().back(), 0.2, 0.01);
  dit.run(cache, both, r0c, rest, &reused);  // the reference is now the 20 % one
  EXPECT_TRUE(reused);

  // Fewer rows (CFG truncation drops uncond): the cond row's residuals still apply.
  const std::vector<Row> cond_only = {{cond, {}}};
  dit.run(cache, cond_only, r0c, rest, &reused);
  EXPECT_TRUE(reused);
  // A new row (an SLG row entering, a changed skip set) has no residuals yet: every block runs.
  SkipSet skip5;
  skip5.set(5);
  const std::vector<Row> with_slg = {{cond, {}}, {uncond, skip22}, {cond, skip5}};
  dit.run(cache, with_slg, r0c, rest, &reused);
  EXPECT_FALSE(reused);
  dit.run(cache, with_slg, r0c, rest, &reused);
  EXPECT_TRUE(reused);
  const std::vector<Row> changed = {{cond, {}}, {uncond, {}}};
  dit.run(cache, changed, r0c, rest, &reused);
  EXPECT_FALSE(reused);
  EXPECT_EQ(cache.calls(), 9);
  EXPECT_EQ(cache.reused(), 5);
  // invalidate() (the pipeline's, when a control input switches between steps): the next call runs in full although
  // block 0's residual did not change at all; the one after reuses again.
  dit.run(cache, changed, r0c, rest, &reused);
  EXPECT_TRUE(reused);
  cache.invalidate();
  dit.run(cache, changed, r0c, rest, &reused);
  EXPECT_FALSE(reused);
  EXPECT_TRUE(std::isnan(cache.diffs().back()));
  dit.run(cache, changed, r0c, rest, &reused);
  EXPECT_TRUE(reused);
  // A row absent from a call loses its residuals: uncond (absent while cond ran alone) comes back in a full call.
  dit.run(cache, cond_only, r0c, rest, &reused);
  EXPECT_TRUE(reused);
  dit.run(cache, changed, r0c, rest, &reused);
  EXPECT_FALSE(reused);
  EXPECT_THROW(accel::FirstBlockCache(arena, 2, 4, 0, 0.1), std::invalid_argument);
  EXPECT_THROW(accel::FirstBlockCache(arena, 2, 4, 2, 0.0), std::invalid_argument);
}

// ---- GPU: the real DiT ----

// The step-cache seam as a deterministic cancel point: delegates to a FirstBlockCache and sets the cancel flag in
// after_first_block of call `at` (after the cache's decision, before the DiT acts on it).
class CancellingHook final : public mi::moss_dit::StepCacheHook {
 public:
  CancellingHook(accel::FirstBlockCache& cache, std::atomic<bool>& flag, int at, bool& reused_at)
      : cache_(cache), flag_(flag), at_(at), reused_at_(reused_at) {}
  void before_blocks(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets,
                     std::int64_t video, cudaStream_t stream) override {
    cache_.before_blocks(x, rows, offsets, video, stream);
  }
  bool after_first_block(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets,
                         std::int64_t video, cudaStream_t stream) override {
    const bool reuse = cache_.after_first_block(x, rows, offsets, video, stream);
    if (++calls_ == at_) {
      reused_at_ = reuse;
      flag_.store(true);
    }
    return reuse;
  }
  void after_blocks(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets, std::int64_t video,
                    bool reused, cudaStream_t stream) override {
    cache_.after_blocks(x, rows, offsets, video, reused, stream);
  }

 private:
  accel::FirstBlockCache& cache_;
  std::atomic<bool>& flag_;
  int at_, calls_ = 0;
  bool& reused_at_;
};

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}

class GpuAccelDit : public ::testing::Test {
 public:
  static std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = a.size() == b.size() ? 0 : std::max(a.size(), b.size());
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
  }

 protected:
  static constexpr std::int64_t kH = 16, kW = 16, kCond = 24, kUncond = 12;
  static constexpr int kSteps = 8;

  static void SetUpTestSuite() {
    const fs::path bundle = weights_root() / "base";
    if (!fs::is_regular_file(bundle / "moss.mossidx")) return;
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle / "moss.mossidx", bundle));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
  }
  static void TearDownTestSuite() {
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!index_) GTEST_SKIP() << "weights not present (" << weights_root() << ")";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  struct Model {
    std::unique_ptr<mi::runtime::DeviceArena> arena;
    std::unique_ptr<MossDit> dit;
  };
  static mi::pipeline::Guidance guidance() {
    mi::pipeline::Guidance g;
    g.cfg = 3.0;
    g.uncond_skip.set(22);
    return g;
  }
  // The largest loop of these tests: CFG + SLG rows, the sampler with the most state.
  static mi::pipeline::Guidance sizing_guidance() {
    mi::pipeline::Guidance g = guidance();
    g.slg_layers.set(5);
    g.slg_scale = 1.0;
    return g;
  }
  Model make_model(const mi::moss_dit::Residency& res, mi::moss_dit::BlockPath path) {
    Model m;
    const std::size_t stack = 2 * mi::runtime::align_up(std::size_t{kCond} * mi::moss_dit::kDim * 2) +
                              std::max(MossDit::encode_text_workspace_bytes(kCond),
                                       mi::runtime::align_up(std::size_t{128} * kH * kW * 4) +
                                           mi::pipeline::denoise_workspace_bytes(kH, kW, kCond, kUncond, sizing_guidance(),
                                                                                 mi::pipeline::SamplerKind::kErSde, path) +
                                           accel::FirstBlockCache::workspace_bytes(kH, kW, 3)) +
                              MossDit::stream_workspace_bytes(*index_, res);
    m.arena = std::make_unique<mi::runtime::DeviceArena>(MossDit::persistent_bytes(*index_, res) + stack);
    m.dit = std::make_unique<MossDit>(*index_, *m.arena, *streams_, false, nullptr, res);
    m.dit->set_block_path(path);
    return m;
  }

  struct Run {
    std::vector<float> latents;
    int calls = 0, reused = 0;
    std::vector<double> diffs;  // per DiT call (NaN: ran in full without a comparison)
    int finished = 0;           // steps finished
    bool cancelled = false;     // moss_dit::Cancelled ended the run
    bool cancel_call_reused = false;  // the call that set the flag (cancel_at) reused the cache
  };
  // An 8-step euler run (moss schedule, CFG 3) on refined random text; `threshold` > 0 sets a FirstBlockCache.
  // cancel_at > 0 (with a cache): the DiT's cancel flag is set from the step-cache seam inside DiT call `cancel_at`
  // (1-based), right after block 0 and its cache decision: before the reuse's stop_early(), or before block 1.
  Run run(Model& m, const mi::pipeline::Guidance& g, double threshold, int stop_after = kSteps,
          mi::pipeline::SamplerKind sampler = mi::pipeline::SamplerKind::kEuler, int cancel_at = 0) {
    mi::runtime::StackScope scope(*m.arena);
    const auto refine = [&](std::int64_t s, std::uint32_t seed) {
      const Tensor out = m.arena->alloc_tensor(mi::DType::kBF16, {s, mi::moss_dit::kDim});
      mi::runtime::StackScope inner(*m.arena);
      const Tensor te = m.arena->alloc_tensor(mi::DType::kBF16, {s, mi::moss_dit::kTextDim});
      const auto h = random_bf16(static_cast<std::size_t>(te.numel()), seed, 2.0f);
      MI_CUDA_CHECK(cudaMemcpyAsync(te.data, h.data(), te.nbytes(), cudaMemcpyHostToDevice, stream()));
      m.dit->encode_text(te, out, stream());
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      return out;
    };
    const Tensor cond = refine(kCond, 11), uncond = refine(kUncond, 12);
    const Tensor latents = m.arena->alloc_tensor(mi::DType::kF32, {128, kH, kW});
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> noise(static_cast<std::size_t>(latents.numel()));
    for (float& v : noise) v = nd(rng);
    MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, noise.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream()));
    Run out;
    std::unique_ptr<accel::FirstBlockCache> cache;
    std::atomic<bool> flag{false};
    std::unique_ptr<CancellingHook> cancelling;
    if (threshold > 0) {
      cache = std::make_unique<accel::FirstBlockCache>(*m.arena, kH, kW, 3, threshold);
      m.dit->set_step_cache(cache.get());
      if (cancel_at > 0) {
        cancelling = std::make_unique<CancellingHook>(*cache, flag, cancel_at, out.cancel_call_reused);
        m.dit->set_step_cache(cancelling.get());
        m.dit->set_cancel(&flag);
      }
    }
    const mi::pipeline::Schedule sched = mi::pipeline::moss_schedule(kSteps, 5.0);
    try {
      const int done = mi::pipeline::denoise(*m.dit, *m.arena, cond, uncond, sched, g, sampler, latents, stream(),
                                             [&](const mi::pipeline::StepInfo& s) {
                                               out.finished = s.step;
                                               return s.step < stop_after;
                                             });
      EXPECT_EQ(done, stop_after);
    } catch (const mi::moss_dit::Cancelled&) {
      out.cancelled = true;
    }
    m.dit->set_step_cache(nullptr);
    m.dit->set_cancel(nullptr);
    out.latents.resize(static_cast<std::size_t>(latents.numel()));
    MI_CUDA_CHECK(cudaMemcpy(out.latents.data(), latents.data, latents.nbytes(), cudaMemcpyDeviceToHost));
    if (cache) out.calls = cache->calls(), out.reused = cache->reused(), out.diffs = cache->diffs();
    return out;
  }
  static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) ab += double(a[i]) * b[i], aa += double(a[i]) * a[i], bb += double(b[i]) * b[i];
    return ab / std::sqrt(aa * bb);
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
};
std::unique_ptr<mi::weights::ModelIndex> GpuAccelDit::index_;
std::unique_ptr<mi::runtime::StreamSet> GpuAccelDit::streams_;

// Off is exact: a cache that never reuses (threshold below any change) leaves every bit of the loop unchanged, as does
// cfg_truncation 1; on the resident fused path, with CUDA graphs, and streamed (where a reuse ends the pass early).
// With reuses the run stays close and gives the same bits on every path.
TEST_F(GpuAccelDit, OffIsBitIdenticalAndReuseIsPathIndependent) {
  const mi::pipeline::Guidance g = guidance();
  constexpr double kNever = 1e-12, kOften = 0.5;
  Run plain, never, often;
  {
    Model m = make_model({}, mi::moss_dit::BlockPath::kFused);
    plain = run(m, g, 0);
    mi::pipeline::Guidance g1 = g;
    g1.cfg_truncation = 1.0;
    EXPECT_EQ(mismatches(run(m, g1, 0).latents, plain.latents), 0u) << "cfg_truncation 1";
    never = run(m, g, kNever);
    EXPECT_EQ(mismatches(never.latents, plain.latents), 0u) << "cache that never reuses";
    EXPECT_EQ(never.calls, kSteps);
    EXPECT_EQ(never.reused, 0);
    often = run(m, g, kOften);
    EXPECT_EQ(often.calls, kSteps);
    EXPECT_GT(often.reused, 0);
    EXPECT_GT(mismatches(often.latents, plain.latents), 0u);
    const double cos = cosine(often.latents, plain.latents);
    std::printf("[accel] fbcache %.2f: %d of %d calls reused, latent cosine vs plain %.5f\n", kOften, often.reused,
                often.calls, cos);
    EXPECT_GT(cos, 0.5);  // a sanity bound: half of an 8-step run reused is far beyond any useful threshold
    EXPECT_EQ(mismatches(run(m, g, kOften).latents, often.latents), 0u) << "deterministic";
  }
  {
    Model m = make_model({}, mi::moss_dit::BlockPath::kFusedGraphs);
    EXPECT_EQ(mismatches(run(m, g, 0).latents, plain.latents), 0u) << "graphs";
    EXPECT_EQ(mismatches(run(m, g, kNever).latents, plain.latents), 0u) << "graphs, never";
    const Run r = run(m, g, kOften);
    EXPECT_EQ(r.reused, often.reused);
    EXPECT_EQ(mismatches(r.latents, often.latents), 0u) << "graphs, reuse";
    EXPECT_GT(m.dit->graph_stats().replays, 0u);
  }
  for (const mi::moss_dit::StreamSource src : {mi::moss_dit::StreamSource::kFile, mi::moss_dit::StreamSource::kHostPinned}) {
    mi::moss_dit::Residency res;
    res.resident_blocks = 21;
    res.source = src;
    const std::string name = src == mi::moss_dit::StreamSource::kFile ? "streamed (file)" : "streamed (host pinned)";
    Model m = make_model(res, mi::moss_dit::BlockPath::kFused);
    EXPECT_EQ(mismatches(run(m, g, kNever).latents, plain.latents), 0u) << name << ", never";
    const Run r = run(m, g, kOften);
    EXPECT_EQ(r.reused, often.reused) << name;
    EXPECT_EQ(mismatches(r.latents, often.latents), 0u) << name << ", reuse";
    EXPECT_EQ(mismatches(run(m, g, 0).latents, plain.latents), 0u) << name << " after reuses";
  }
  {
    // Every block streamed, block 0 included: a reuse stops the pass after the first streamed block.
    mi::moss_dit::Residency res;
    res.resident_blocks = 0;
    Model m = make_model(res, mi::moss_dit::BlockPath::kFused);
    EXPECT_EQ(mismatches(run(m, g, kNever).latents, plain.latents), 0u) << "fully streamed, never";
    const Run r = run(m, g, kOften);
    EXPECT_EQ(r.reused, often.reused);
    EXPECT_EQ(mismatches(r.latents, often.latents), 0u) << "fully streamed, reuse";
    EXPECT_EQ(mismatches(run(m, g, 0).latents, plain.latents), 0u) << "fully streamed after reuses";
  }
}

// Guidance windows on the real DiT: the uncond row's skip set changes and the SLG row enters and leaves at the window's
// edges (steps 2 and 6 of 8); those calls always run in full. Heun (two calls per step, the second at the next sigma)
// caches per call. A cache that never reuses is exact in both.
TEST_F(GpuAccelDit, GuidanceWindowsAndHeun) {
  Model m = make_model({}, mi::moss_dit::BlockPath::kFused);
  mi::pipeline::Guidance g = sizing_guidance();
  g.slg_start = 0.25;
  g.slg_stop = 0.75;
  const Run plain = run(m, g, 0);
  EXPECT_EQ(mismatches(run(m, g, 1e-12).latents, plain.latents), 0u);
  const Run r = run(m, g, 1.0);
  ASSERT_EQ(r.diffs.size(), static_cast<std::size_t>(kSteps));
  EXPECT_GT(r.reused, 0);
  for (const int edge : {0, 2, 6}) EXPECT_TRUE(std::isnan(r.diffs[static_cast<std::size_t>(edge)])) << "call " << edge;
  for (const int inside : {1, 3, 7}) EXPECT_FALSE(std::isnan(r.diffs[static_cast<std::size_t>(inside)])) << "call " << inside;
  std::printf("[accel] SLG window [0.25, 0.75): %d of %d calls reused\n", r.reused, r.calls);

  const mi::pipeline::Guidance gh = guidance();
  const Run heun = run(m, gh, 0, kSteps, mi::pipeline::SamplerKind::kHeun);
  const Run heun_never = run(m, gh, 1e-12, kSteps, mi::pipeline::SamplerKind::kHeun);
  EXPECT_EQ(mismatches(heun_never.latents, heun.latents), 0u);
  EXPECT_EQ(heun_never.calls, 2 * kSteps - 1);  // no corrector into sigma 0
  const Run heun_often = run(m, gh, 0.5, kSteps, mi::pipeline::SamplerKind::kHeun);
  EXPECT_EQ(heun_often.calls, 2 * kSteps - 1);
  EXPECT_GT(heun_often.reused, 0);
  for (const float v : heun_often.latents) ASSERT_TRUE(std::isfinite(v));
  std::printf("[accel] heun: %d of %d calls reused, latent cosine vs plain %.5f\n", heun_often.reused, heun_often.calls,
              cosine(heun_often.latents, heun.latents));
}

// A cancel inside a cached pass, from the step-cache seam, on streamed DiTs (file and pinned host copies): set in a
// call that reuses the cache, it races the readers against stop_early() and that call still finishes (the next call
// throws at block 0); set in a full call, block 1 throws. Either way nothing is left behind: the plain run and a fresh
// cached run afterwards are bit-identical to the ones before.
TEST_F(GpuAccelDit, CancelFromTheCacheSeamLeavesNoState) {
  const mi::pipeline::Guidance g = guidance();
  constexpr double kOften = 0.5;
  for (const mi::moss_dit::StreamSource src : {mi::moss_dit::StreamSource::kFile, mi::moss_dit::StreamSource::kHostPinned}) {
    mi::moss_dit::Residency res;
    res.resident_blocks = 21;
    res.source = src;
    const char* name = src == mi::moss_dit::StreamSource::kFile ? "file" : "host_pinned";
    Model m = make_model(res, mi::moss_dit::BlockPath::kFused);
    const Run plain = run(m, g, 0), often = run(m, g, kOften);
    ASSERT_EQ(often.diffs.size(), static_cast<std::size_t>(kSteps));
    int reused_call = 0, full_call = 0;  // 1-based, before the last call (the run must have a next call)
    for (int i = 1; i < kSteps - 1; ++i) {
      const bool reused = often.diffs[static_cast<std::size_t>(i)] < kOften;
      if (reused && reused_call == 0) reused_call = i + 1;
      if (!reused && full_call == 0) full_call = i + 1;
    }
    ASSERT_GT(reused_call, 0) << name;
    if (full_call == 0) full_call = 1;  // the first call always runs in full
    for (const int at : {reused_call, full_call}) {
      const Run c = run(m, g, kOften, kSteps, mi::pipeline::SamplerKind::kEuler, at);
      EXPECT_TRUE(c.cancelled) << name << " call " << at;
      EXPECT_EQ(c.cancel_call_reused, at == reused_call) << name << " call " << at;
      EXPECT_EQ(c.finished, at == reused_call ? at : at - 1) << name << " call " << at;
      EXPECT_EQ(mismatches(run(m, g, 0).latents, plain.latents), 0u) << name << ": plain after a cancel at call " << at;
      const Run again = run(m, g, kOften);
      EXPECT_EQ(again.reused, often.reused) << name;
      EXPECT_EQ(mismatches(again.latents, often.latents), 0u) << name << ": cached after a cancel at call " << at;
    }
    std::printf("[accel] %s: cancels at reused call %d and full call %d left no state\n", name, reused_call, full_call);
  }
}

// CFG truncation: the steps before the truncation point are the CFG run's; truncation 0 is the cond row alone, i.e.
// the run without CFG.
TEST_F(GpuAccelDit, CfgTruncationRunsCondAloneAfterThePoint) {
  Model m = make_model({}, mi::moss_dit::BlockPath::kFused);
  const mi::pipeline::Guidance g = guidance();
  mi::pipeline::Guidance half = g;
  half.cfg_truncation = 0.5;  // steps 4..7 without the uncond row
  EXPECT_EQ(mismatches(run(m, half, 0, 4).latents, run(m, g, 0, 4).latents), 0u);
  EXPECT_GT(mismatches(run(m, half, 0, 5).latents, run(m, g, 0, 5).latents), 0u);  // step index 4 is the first cond-only
  const Run full = run(m, g, 0), truncated = run(m, half, 0);
  EXPECT_GT(mismatches(truncated.latents, full.latents), 0u);
  std::printf("[accel] cfg_truncation 0.5: latent cosine vs full CFG %.5f\n", cosine(truncated.latents, full.latents));
  mi::pipeline::Guidance none = g, zero = g;
  none.cfg = 1.0;
  zero.cfg_truncation = 0.0;
  EXPECT_EQ(mismatches(run(m, zero, 0).latents, run(m, none, 0).latents), 0u);
  // With the step cache: the uncond row leaving keeps the cond row's residuals usable.
  const Run both = run(m, half, 0.5);
  EXPECT_GT(both.reused, 0);
}

// The whole pipeline on an INT8 DiT file (High tier) with a runtime LoRA: accelerator options at their off values give
// the default request's bits; a cache that never reuses changes nothing either; CFG truncation and reuses are counted
// and reported.
TEST(GpuAccelPipeline, Int8LoraOffIsExactOnIsReported) {
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  const fs::path peft = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(int8) || !fs::is_regular_file(peft)) {
    GTEST_SKIP() << "SFT bundle, INT8 DiT or LoRA file not present";
  }
  mi::pipeline::PipelineOptions options;
  options.dit_file = int8;
  options.tier = mi::planner::Tier::kHigh;
  mi::pipeline::Pipeline pipe(options);
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "portrait of a woman in a garden, soft light";
  req.height = req.width = 256;
  req.steps = 10;
  req.decode = false;
  req.loras = {{peft, 0.6f}};
  const mi::pipeline::Result base = pipe.generate(req);
  EXPECT_EQ(base.accel.dit_calls, 0);
  EXPECT_EQ(base.accel.cfg_truncated_steps, 0);

  mi::pipeline::Request off = req;
  off.accel.cfg_truncation = 1.0;
  off.accel.step_cache = accel::StepCacheKind::kOff;
  off.accel.step_cache_threshold = 0.3;  // unused without a cache
  EXPECT_EQ(GpuAccelDit::mismatches(pipe.generate(off).latents, base.latents), 0u);

  mi::pipeline::Request never = req;
  never.accel.step_cache = accel::StepCacheKind::kFbCache;
  never.accel.step_cache_threshold = 1e-12;
  const mi::pipeline::Result nr = pipe.generate(never);
  EXPECT_EQ(GpuAccelDit::mismatches(nr.latents, base.latents), 0u);
  EXPECT_EQ(nr.accel.dit_calls, 10);
  EXPECT_EQ(nr.accel.cached_calls, 0);
  EXPECT_NE(nr.tier.reason.find("lossy accelerators"), std::string::npos);

  mi::pipeline::Request on = req;
  on.accel.cfg_truncation = 0.8;
  on.accel.step_cache = accel::StepCacheKind::kFbCache;
  on.accel.step_cache_threshold = 0.3;
  const mi::pipeline::Result r = pipe.generate(on);
  EXPECT_EQ(r.accel.cfg_truncated_steps, 2);
  EXPECT_EQ(r.accel.dit_calls, 10);
  EXPECT_GT(r.accel.cached_calls, 0);
  EXPECT_EQ(static_cast<int>(r.accel.cached_steps.size()), r.accel.cached_calls);
  EXPECT_GT(GpuAccelDit::mismatches(r.latents, base.latents), 0u);
  for (const float v : r.latents) ASSERT_TRUE(std::isfinite(v));
  EXPECT_NE(r.tier.reason.find("lossy accelerators"), std::string::npos);
  std::printf("[accel pipeline] int8 + LoRA, 10 steps: fbcache 0.3 reused %d of %d calls (steps", r.accel.cached_calls,
              r.accel.dit_calls);
  for (const int s : r.accel.cached_steps) std::printf(" %d", s);
  std::printf("), denoise %.2f s vs %.2f s plain (dev)\n", r.timings.denoise_s, base.timings.denoise_s);
  const mi::json::Value rep = mi::json::parse("{" + mi::pipeline::report_fields(on, r, "", "") + "}");
  EXPECT_TRUE(rep.at("accelerators").at("lossy").as_bool());
  EXPECT_EQ(rep.at("models").at("dit").at("tier").as_string(), "high");
}

std::size_t count_of(const std::string& s, const std::string& what) {
  std::size_t n = 0;
  for (std::size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
}

// Each tier with accelerators: resolved as without them (the bf16 bundle stays Reference, the INT8 file keeps High and
// its 8-bit attention, the INT4 file runs Turbo), with one accelerator remark. On the INT4 DiT a cache that never reuses
// is exact, and reuses run.
TEST(GpuAccelPipeline, TiersIgnoreAcceleratorsAndInt4Caches) {
  using namespace mi::pipeline;
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  const fs::path int4 = "/workspace/Models/community/toxicdog/moss_image_sft_int4_convrot.safetensors";
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(int8) || !fs::is_regular_file(int4)) {
    GTEST_SKIP() << "SFT bundle, INT8 or INT4 DiT not present";
  }
  Request req;
  req.model = weights_root() / "sft";
  req.prompt = "portrait of a woman in a garden, soft light";
  req.height = req.width = 256;
  req.steps = 8;
  req.decode = false;
  Request fast = req;
  fast.accel.step_cache = accel::StepCacheKind::kFbCache;
  fast.accel.step_cache_threshold = 0.5;
  fast.accel.cfg_truncation = 0.8;
  const auto check = [&](const Result& off, const Result& on, const char* what) {
    EXPECT_EQ(on.tier.tier, off.tier.tier) << what;
    EXPECT_EQ(on.tier.attention, off.tier.attention) << what;
    EXPECT_EQ(on.tier.precision, off.tier.precision) << what;
    EXPECT_EQ(count_of(on.tier.reason, "accelerator"), 1u) << what << ": " << on.tier.reason;
    EXPECT_EQ(count_of(off.tier.reason, "accelerator"), 0u) << what << ": " << off.tier.reason;
    std::printf("[accel tier] %s: %s / %s, reason: %s\n", what, mi::planner::tier_name(on.tier.tier),
                mi::planner::attention_name(on.tier.attention), on.tier.reason.c_str());
  };
  {
    Pipeline pipe;
    const Result off = pipe.generate(req), on = pipe.generate(fast);
    EXPECT_EQ(off.tier.tier, mi::planner::Tier::kReference);
    check(off, on, "bf16 bundle");
  }
  {
    PipelineOptions o;
    o.dit_file = int8;
    o.tier = mi::planner::Tier::kHigh;
    o.embedding_cache = false;
    Pipeline pipe(o);
    const Result off = pipe.generate(req), on = pipe.generate(fast);
    EXPECT_EQ(off.tier.tier, mi::planner::Tier::kHigh);
    EXPECT_EQ(off.tier.attention, mi::moss_dit::attention_supported(mi::moss_dit::AttentionMode::kQ8Fp8)
                                      ? mi::planner::AttentionMode::kQ8Fp8
                                      : mi::planner::AttentionMode::kQ8Fp16);
    check(off, on, "int8 file");
  }
  {
    PipelineOptions o;
    o.dit_file = int4;
    o.tier = mi::planner::Tier::kTurbo;
    o.embedding_cache = false;
    Pipeline pipe(o);
    const Result off = pipe.generate(req);
    EXPECT_EQ(off.tier.tier, mi::planner::Tier::kTurbo);
    Result on;
    ASSERT_NO_THROW(on = pipe.generate(fast));
    check(off, on, "int4 file");
    EXPECT_GT(on.accel.cached_calls, 0);
    Request never = req;
    never.accel.step_cache = accel::StepCacheKind::kFbCache;
    never.accel.step_cache_threshold = 1e-12;
    const Result nr = pipe.generate(never);
    EXPECT_EQ(GpuAccelDit::mismatches(nr.latents, off.latents), 0u) << "INT4, never";
    EXPECT_EQ(nr.accel.dit_calls, 8);
  }
}

// The hires fix's second pass runs with its own cache (steps count on across passes); a cancel between the steps of a
// cached pass leaves nothing behind: the next request without accelerators is bit-identical to the one before.
TEST(GpuAccelPipeline, HiresPassAndCancelLeaveNoState) {
  using namespace mi::pipeline;
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx")) GTEST_SKIP() << "SFT bundle not present";
  Pipeline pipe;
  Request req;
  req.model = weights_root() / "sft";
  req.prompt = "a lighthouse on a cliff at dusk, photo";
  req.height = req.width = 256;
  req.steps = 6;
  req.hires.width = req.hires.height = 384;
  req.hires.strength = 0.5;  // 3 of 6 second-pass steps
  req.decode = false;
  const Result plain = pipe.generate(req);
  ASSERT_EQ(plain.steps_run, 9);
  Request never = req;
  never.accel.step_cache = accel::StepCacheKind::kFbCache;
  never.accel.step_cache_threshold = 1e-12;
  const Result nr = pipe.generate(never);
  EXPECT_EQ(GpuAccelDit::mismatches(nr.latents, plain.latents), 0u);
  EXPECT_EQ(nr.accel.dit_calls, 9);
  Request often = req;
  often.accel.step_cache = accel::StepCacheKind::kFbCache;
  often.accel.step_cache_threshold = 1.0;
  const Result r = pipe.generate(often);
  EXPECT_EQ(r.accel.dit_calls, 9);
  // Pass 2's first call (step 7) has nothing to compare with; its later calls reuse pass 2's own residuals.
  EXPECT_EQ(std::count(r.accel.cached_steps.begin(), r.accel.cached_steps.end(), 7), 0);
  EXPECT_GT(std::count_if(r.accel.cached_steps.begin(), r.accel.cached_steps.end(), [](int s) { return s > 7; }), 0);
  for (const float v : r.latents) ASSERT_TRUE(std::isfinite(v));

  Request simple = req;
  simple.hires = {};
  simple.steps = 12;
  const Result before = pipe.generate(simple);
  Request cached = simple;
  cached.accel = often.accel;
  // Between steps: stop after step 5.
  std::atomic<bool> cancel{false};
  GenerateControl stop;
  stop.cancel = &cancel;
  stop.on_step = [&](const StepInfo& s) {
    if (s.step == 5) cancel.store(true);
    return true;
  };
  const Result c1 = pipe.generate(cached, stop);
  EXPECT_TRUE(c1.cancelled);
  EXPECT_EQ(c1.steps_run, 5);
  EXPECT_EQ(GpuAccelDit::mismatches(pipe.generate(simple).latents, before.latents), 0u) << "after a cancel between steps";
  // A cancel inside a forward (racing the streaming pass and stop_early) is pinned deterministically at the DiT level:
  // GpuAccelDit.CancelFromTheCacheSeamLeavesNoState.
}

// A control input's step window opening or closing changes the DiT's input embedding between steps; at a low control
// strength block 0's relative change can stay below any threshold. The step right after a window boundary always runs
// every block (FirstBlockCache::invalidate from the pipeline's step callback), while without a boundary the same step
// is served from the cache at this threshold.
TEST(GpuAccelPipeline, ControlWindowBoundaryIsNeverCached) {
  const fs::path adapter = "/workspace/Models/community/comfy-org/moss_image_color_depth_control.safetensors";
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(adapter)) {
    GTEST_SKIP() << "SFT bundle or control adapter not present";
  }
  auto depth = std::make_shared<mi::pipeline::Image>();
  depth->width = depth->height = 256;
  depth->channels = 1;
  depth->pixels.resize(256 * 256);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      const double dx = (x - 128) / 77.0, dy = (y - 128) / 77.0;
      depth->pixels[static_cast<std::size_t>(y) * 256 + static_cast<std::size_t>(x)] = dx * dx + dy * dy < 1 ? 230 : 25;
    }
  }
  mi::pipeline::Pipeline pipe;
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "a red ball on a wooden table, photo";
  req.width = req.height = 256;
  req.steps = 10;
  req.seed = 7;
  req.decode = false;
  req.control.adapter = adapter;
  mi::pipeline::Request::ControlImage c;
  c.kind = "depth";
  c.image = depth;
  c.strength = 0.05;  // weak: the input change alone would not stop a reuse
  req.control.images.push_back(c);
  req.accel.step_cache = accel::StepCacheKind::kFbCache;
  req.accel.step_cache_threshold = 1.0;  // reuse whenever allowed
  const auto cached = [](const mi::pipeline::Result& r, int step) {
    return std::find(r.accel.cached_steps.begin(), r.accel.cached_steps.end(), step) != r.accel.cached_steps.end();
  };
  const auto show = [](const char* what, const mi::pipeline::Result& r) {
    std::printf("[accel control] %-22s cached steps:", what);
    for (const int s : r.accel.cached_steps) std::printf(" %d", s);
    std::printf("\n");
  };
  // No boundary: steps 6 and 4 (1-based) are reused.
  const mi::pipeline::Result whole = pipe.generate(req);
  show("window [0, 1)", whole);
  EXPECT_TRUE(cached(whole, 6));
  EXPECT_TRUE(cached(whole, 4));
  // Closing after half the steps: step 6 (index 5, the first without control) runs in full.
  mi::pipeline::Request closes = req;
  closes.control.images[0].end = 0.5;
  const mi::pipeline::Result rc = pipe.generate(closes);
  show("window [0, 0.5)", rc);
  EXPECT_FALSE(cached(rc, 6));
  EXPECT_GT(rc.accel.cached_calls, 0);
  // Opening at 30 %: step 4 (index 3, the first with control) runs in full.
  mi::pipeline::Request opens = req;
  opens.control.images[0].start = 0.3;
  const mi::pipeline::Result ro = pipe.generate(opens);
  show("window [0.3, 1)", ro);
  EXPECT_FALSE(cached(ro, 4));
  EXPECT_GT(ro.accel.cached_calls, 0);
}

}  // namespace
