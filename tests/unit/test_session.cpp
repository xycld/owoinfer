// pipeline/session.h, the component session (design §7.1 / §7.2, P1 part).
//   Session.*     CPU: Session::validate, the checks begin_run makes before any work (rows, latents, guidance modes,
//                 the schedule, the step cache, mi_generate's shared rules, the token cap).
//   GpuSession.*  on real weights (GTEST_SKIP when absent): an L = 1 MOSS-layout run plans exactly as mi_generate's
//                 equivalent request (§7.2: the plan JSON) and, driven by the recipe's euler-on-moss host loop through the
//                 staging, gives mi_generate's final latents bit for bit; the state errors of a session on its gate;
//                 table A's other host loops (HOST CFG, MOSS rows, GUIDED_DENOISED on simple, a latent batch of 2)
//                 against mi_generate's final latents (the internal hook: decode off).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/session.h"
#include "runtime/job_gate.h"

namespace {

namespace fs = std::filesystem;
using mi::pipeline::Conditioning;
using mi::pipeline::GuidanceMode;
using mi::pipeline::RunSpec;
using mi::pipeline::Session;
using mi::pipeline::SessionOptions;
using mi::pipeline::StepOutput;

std::shared_ptr<const Conditioning> cond(std::int64_t tokens, bool engine = true) {
  auto e = std::make_shared<mi::pipeline::Embedding>();
  e->hidden = mi::moss_dit::kTextDim;
  e->data.assign(static_cast<std::size_t>(tokens) * mi::moss_dit::kTextDim, 0);
  auto c = std::make_shared<Conditioning>();
  c->embedding = std::move(e);
  c->engine_encoded = engine;
  return c;
}

RunSpec host_spec(int rows) {
  RunSpec s;
  s.width = s.height = 512;
  for (int k = 0; k < rows; ++k) s.rows.push_back({cond(20 + k), {}, 0});
  return s;
}

SessionOptions options(int max_rows = 3, int max_latents = 1) {
  SessionOptions o;
  o.max_rows = max_rows;
  o.max_latents = max_latents;
  return o;
}

void refused(const RunSpec& s, const SessionOptions& o, const std::string& fragment) {
  try {
    Session::validate(s, o);
    ADD_FAILURE() << "accepted; expected a refusal containing \"" << fragment << "\"";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
  }
}

TEST(Session, AcceptsHostAndMossRuns) {
  EXPECT_NO_THROW(Session::validate(host_spec(3), options()));
  RunSpec m = host_spec(2);
  m.guidance = GuidanceMode::kMoss;
  m.output = StepOutput::kGuidedVelocity;
  m.rows[1].skip_layers = {22};
  m.slg_layers = {7, 8};
  m.slg_scale = 2.0;
  m.cfg_truncation = 0.8;
  EXPECT_NO_THROW(Session::validate(m, options()));
  RunSpec batch = host_spec(4);
  batch.rows[2].latent = batch.rows[3].latent = 1;
  EXPECT_NO_THROW(Session::validate(batch, options(4, 2)));
  RunSpec sig = host_spec(1);
  sig.sigmas = {1.0f, 0.5f, 0.5f, 0.0f};
  EXPECT_NO_THROW(Session::validate(sig, options()));
}

TEST(Session, RefusesBadRows) {
  refused(host_spec(4), options(3), "1 to max_rows (3) rows, not 4");
  RunSpec none;
  refused(none, options(), "rows, not 0");
  RunSpec null_cond = host_spec(1);
  null_cond.rows[0].conditioning = nullptr;
  refused(null_cond, options(), "rows[0]: no conditioning");
  RunSpec long_text = host_spec(1);
  long_text.rows[0].conditioning = cond(4097);
  refused(long_text, options(), "[1..4096, 4096]");
  RunSpec skips = host_spec(1);
  skips.rows[0].skip_layers = {22, 22};
  refused(skips, options(), "nonnegative and unique");
  skips.rows[0].skip_layers = {42};
  refused(skips, options(), "skip_layers must contain integer block indices in [0, 42)");
  RunSpec latent = host_spec(2);
  latent.rows[1].latent = 1;
  refused(latent, options(3, 1), "outside the session's 0..0");
  latent.rows[1].latent = 2;
  refused(latent, options(3, 3), "no row reads latent 1");
}

TEST(Session, RefusesGuidanceMismatches) {
  RunSpec m = host_spec(3);
  m.guidance = GuidanceMode::kMoss;
  refused(m, options(), "MOSS guidance takes 1 or 2 rows");
  RunSpec ml = host_spec(2);
  ml.guidance = GuidanceMode::kMoss;
  ml.rows[1].latent = 1;
  refused(ml, options(3, 2), "on latent 0");
  RunSpec cond_skip = host_spec(2);
  cond_skip.guidance = GuidanceMode::kMoss;
  cond_skip.rows[0].skip_layers = {5};
  refused(cond_skip, options(), "rows[0] is the cond row, which skips no blocks");
  RunSpec g1 = host_spec(2);
  g1.guidance = GuidanceMode::kMoss;
  g1.output = StepOutput::kGuidedDenoised;
  g1.cfg = 1.0;
  refused(g1, options(), "need cfg != 1");
  RunSpec hg = host_spec(2);
  hg.output = StepOutput::kGuidedVelocity;
  refused(hg, options(), "Guided outputs need MOSS guidance");
  for (int field = 0; field < 4; ++field) {
    RunSpec h = host_spec(2);
    if (field == 0) h.cfg = 4.0;
    if (field == 1) h.slg_scale = 1.0;
    if (field == 2) h.cfg_truncation = 0.5;
    if (field == 3) h.slg_layers = {3};
    refused(h, options(), "HOST guidance keeps cfg");
  }
}

TEST(Session, RefusesScheduleStepCacheAndControlErrors) {
  RunSpec s = host_spec(1);
  s.sigmas = {1.0f};
  refused(s, options(), "");
  s.sigmas = {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
  refused(s, options(), "");
  s.sigmas = {0.5f, 1.0f, 0.0f};
  refused(s, options(), "");
  RunSpec zero = host_spec(1);
  zero.denoise_strength = 0.0;
  refused(zero, options(), "no step left");
  RunSpec strength = host_spec(1);
  strength.denoise_strength = 1.5;
  refused(strength, options(), "Denoise strength must be between 0 and 1");
  RunSpec cache = host_spec(1);
  cache.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;
  refused(cache, options(), "explicit threshold");
  cache.step_cache_threshold = 0.05;
  EXPECT_NO_THROW(Session::validate(cache, options()));
  RunSpec ctl = host_spec(1);
  ctl.control_images.push_back({"depth", nullptr, 1.0, 0.0, 1.0});
  refused(ctl, options(), "Control images need the session's control adapter");
  RunSpec size = host_spec(1);
  size.width = 500;
  refused(size, options(), "Height and width must be positive multiples of 16.");
  RunSpec steps = host_spec(1);
  steps.steps = 0;
  refused(steps, options(), "Steps and image count must be positive.");
}

TEST(Session, TokenCap) {
  // 2048^2: 16384 video tokens per row. Three rows of 4096-token texts carry 61440 <= 61506; four do not.
  RunSpec s;
  s.width = s.height = 2048;
  for (int k = 0; k < 3; ++k) s.rows.push_back({cond(4096), {}, 0});
  EXPECT_NO_THROW(Session::validate(s, options(4)));
  s.rows.push_back({cond(4096), {}, 0});
  refused(s, options(4), "81920 tokens, more than the 61506");
  // The cap is per latent (one forward per latent): the same four rows over two latents carry 40960 each.
  s.rows[2].latent = s.rows[3].latent = 1;
  EXPECT_NO_THROW(Session::validate(s, options(4, 2)));
  // MOSS's largest call (cond, uncond, SLG) stays within the range at any valid size and text length: the self-check
  // validated exactly three rows of 4.2 MP and 4096 tokens.
  RunSpec m;
  m.width = m.height = 2048;
  m.guidance = GuidanceMode::kMoss;
  m.rows = {{cond(4096), {}, 0}, {cond(4096), {22}, 0}};
  m.slg_layers = {7};
  m.slg_scale = 1.0;
  EXPECT_NO_THROW(Session::validate(m, options()));
}

// ---- GPU, real weights

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
const fs::path kInt8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";

mi::pipeline::PipelineOptions pipeline_options() {
  mi::pipeline::PipelineOptions o;
  o.dit_file = kInt8;
  o.tier = mi::planner::Tier::kHigh;
  o.attention = mi::planner::AttentionMode::kQ8Fp8;
  mi::planner::HardwareProfile hw;  // fixed rates: no probe, no learning between the compared calls
  hw.h2d_gbps = 24;
  hw.disk_gbps = 3;
  hw.host_gbps = 18;
  hw.bf16_tflops = 150;
  hw.measured = true;
  o.hardware = hw;
  return o;
}

mi::pipeline::Request moss_request() {
  mi::pipeline::Request q;
  q.model = weights_root() / "sft";
  q.prompt = "a red apple on a wooden table, studio light";
  q.negative_prompt = "blurry";
  q.width = q.height = 512;
  q.steps = 6;
  q.seed = 7;
  return q;
}

// The plan JSON without live machine state, which moves between any two calls: the free device and available host
// memory (the arena cap and the host limit, in the explanation too) and the process's host private bytes.
std::string plan_facts(const mi::planner::ExecutionPlan& p) {
  std::string j = p.to_json();
  j = std::regex_replace(j, std::regex("(free|host private|Arena cap) [0-9.]+ GiB"), "$1 X GiB");
  return std::regex_replace(j, std::regex("\"(host_private_bytes|arena_cap_bytes|host_limit_bytes)\": [0-9]+"), "\"$1\": X");
}

class GpuSession : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kInt8)) {
      GTEST_SKIP() << "SFT bundle or INT8 DiT not present";
    }
  }
};

// §7.2: an L = 1 MOSS-layout run (distinct conditionings for cond and uncond: one refined text each, as mi_generate)
// plans exactly as mi_generate's equivalent request on the same engine state (an embedding-cache hit, sampler euler,
// decode on), and the recipe's euler-on-moss host loop through the staging gives mi_generate's final latents.
TEST_F(GpuSession, MossRunPlansAndSamplesAsGenerate) {
  mi::pipeline::Pipeline pipe(pipeline_options());
  mi::runtime::JobGate gate;
  mi::pipeline::Request q = moss_request();
  pipe.generate(q);  // loads the models and fills the embedding cache: the held state every later call plans against
  const mi::pipeline::Result ref = pipe.generate(q);

  SessionOptions so;
  so.plan_request = q;
  Session session(pipe, gate, so);
  EXPECT_THROW(Session(pipe, gate, so), mi::runtime::JobGate::Busy);
  const std::vector<std::string> prompts = {q.prompt, q.negative_prompt};
  const auto conds = session.encode_texts(prompts, q.max_sequence_length);
  ASSERT_EQ(conds.size(), 2u);
  EXPECT_EQ(conds[0]->length(), ref.cond_tokens);
  EXPECT_EQ(conds[1]->length(), ref.uncond_tokens);

  RunSpec spec;
  spec.width = q.width;
  spec.height = q.height;
  spec.steps = q.steps;
  spec.guidance = GuidanceMode::kMoss;
  spec.output = StepOutput::kGuidedVelocity;  // mi_sampler_prediction(euler, moss)
  spec.exact_sigma = true;
  spec.rows = {{conds[0], {}, 0}, {conds[1], {22}, 0}};
  EXPECT_THROW(session.step({}), mi::runtime::JobGate::Refused);  // no run is open
  session.begin_run(spec);
  EXPECT_THROW(session.begin_run(spec), mi::runtime::JobGate::Busy);
  const mi::pipeline::RunInfo info = session.run_info();
  EXPECT_EQ(info.outputs, 1);
  EXPECT_EQ(info.rows, 3u);
  EXPECT_TRUE(info.schedule_matches);
  EXPECT_EQ(info.steps, q.steps);

  const mi::pipeline::StagingViews v = session.staging();
  const std::vector<float> noise =
      mi::pipeline::initial_noise(q.noise_layout, static_cast<std::uint64_t>(q.seed), 128, info.latent_height, info.latent_width);
  std::vector<float> x = noise;  // sigma_0 = 1: x = noise
  const std::vector<float>& s = session.run_sigmas();
  auto* xs = static_cast<float*>(v.x.data);
  const auto* vel = static_cast<const float*>(v.out[0].data);
  for (int i = 0; i < info.steps; ++i) {
    std::copy(x.begin(), x.end(), xs);
    mi::pipeline::StepArgs a;
    a.sigma = s[static_cast<std::size_t>(i)];
    a.x = v.x;
    a.out[0] = v.out[0];
    const mi::pipeline::StepReport rep = session.step(a);
    EXPECT_EQ(rep.step, i);
    EXPECT_TRUE(rep.on_schedule);
    // SamplerDriver's official euler: h = fp32(double(s[i+1]) - double(s[i])), x = x + h * v (one rounding per op).
    const float h = static_cast<float>(static_cast<double>(s[static_cast<std::size_t>(i) + 1]) -
                                       static_cast<double>(s[static_cast<std::size_t>(i)]));
    for (std::size_t k = 0; k < x.size(); ++k) {
      const float t = h * vel[k];
      x[k] = x[k] + t;
    }
  }
  mi::pipeline::StepArgs bad;
  bad.sigma = 0.37f;  // not a schedule entry: refused under exact_sigma, the run still usable
  bad.x = v.x;
  bad.out[0] = v.out[0];
  EXPECT_THROW(session.step(bad), std::invalid_argument);
  const std::optional<mi::pipeline::RunResult> done = session.end_run();
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(done->calls, q.steps);
  EXPECT_EQ(done->off_schedule_calls, 0);
  EXPECT_TRUE(done->note.empty()) << done->note;
  EXPECT_EQ(done->result.memory.dit_loads, 0);
  EXPECT_EQ(plan_facts(done->result.plan), plan_facts(ref.plan));
  EXPECT_FALSE(session.end_run().has_value());
  ASSERT_EQ(x.size(), ref.latents.size());
  std::size_t diff = 0;
  for (std::size_t k = 0; k < x.size(); ++k) diff += x[k] != ref.latents[k];
  EXPECT_EQ(diff, 0u) << "of " << x.size() << " latent values";
  EXPECT_TRUE(session.close());
  EXPECT_FALSE(session.close());
}

// §7.2's latent-level comparison through the internal hook (Result::latents, decode off), where the C API tests compare
// RGB bytes, which can hide a 1-ulp latent difference: host loops through the staging give mi_generate's final latents
// bit for bit for HOST rows {cond, uncond + {22}} with the host's CFG, for MOSS rows with the host's per-op combine, for
// MOSS GUIDED_DENOISED with euler on simple, and for each image of a HOST latent batch of 2 (two seeds in one run).
TEST_F(GpuSession, TableALatentsEqualGenerate) {
  mi::pipeline::Pipeline pipe(pipeline_options());
  mi::runtime::JobGate gate;
  mi::pipeline::Request q = moss_request();
  q.decode = false;  // the final latents only
  pipe.generate(q);  // loads the models and fills the embedding cache
  const mi::pipeline::Result ref7 = pipe.generate(q);
  mi::pipeline::Request q8 = q;
  q8.seed = 8;
  const mi::pipeline::Result ref8 = pipe.generate(q8);
  mi::pipeline::Request qs = q;
  qs.scheduler = mi::pipeline::SchedulerKind::kSimple;  // euler on simple: the denoised prediction
  const mi::pipeline::Result ref_simple = pipe.generate(qs);
  ASSERT_FALSE(ref7.latents.empty());
  ASSERT_NE(ref7.latents, ref8.latents);

  SessionOptions so;
  so.plan_request = q;
  so.max_rows = 4;
  so.max_latents = 2;
  Session session(pipe, gate, so);
  const std::vector<std::string> prompts = {q.prompt, q.negative_prompt};
  const auto conds = session.encode_texts(prompts, q.max_sequence_length);
  const float g = static_cast<float>(q.cfg);
  const auto noise = [&](std::int64_t seed, std::int64_t h, std::int64_t w) {
    return mi::pipeline::initial_noise(q.noise_layout, static_cast<std::uint64_t>(seed), 128, h, w);
  };
  // x = s0 * noise + (1 - s0) * 0, one rounding per op (flow_noise_scaling without an init image).
  const auto start = [](const std::vector<float>& n, float s0) {
    std::vector<float> x(n.size());
    const float keep = 1.0f - s0;
    for (std::size_t k = 0; k < x.size(); ++k) {
      const float a = s0 * n[k];
      const float b = keep * 0.0f;
      x[k] = a + b;
    }
    return x;
  };
  // CFG in velocity space per op: u + g (c - u).
  const auto cfg = [g](const float* c, const float* u, float* v, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
      const float d = c[k] - u[k];
      const float t = g * d;
      v[k] = u[k] + t;
    }
  };
  // One run of `spec` sampled with euler (velocity, or with `denoised` the denoised form) from noise of `seeds` (one
  // per latent); `predict` turns the staged outputs of a call into the prediction [L, 128, h, w].
  using Predict = std::function<void(const mi::pipeline::StagingViews&, const std::vector<float>& x, float sigma,
                                     std::uint32_t rows, std::vector<float>& pred)>;
  const auto sample = [&](const RunSpec& spec, std::vector<std::int64_t> seeds, bool denoised, const Predict& predict) {
    session.begin_run(spec);
    const mi::pipeline::RunInfo info = session.run_info();
    const mi::pipeline::StagingViews v = session.staging();
    const std::vector<float> s = session.run_sigmas();
    std::vector<float> x;
    for (const std::int64_t seed : seeds) {
      const std::vector<float> xl = start(noise(seed, info.latent_height, info.latent_width), s[0]);
      x.insert(x.end(), xl.begin(), xl.end());
    }
    std::vector<float> pred;
    for (int i = 0; i < info.steps; ++i) {
      const float si = s[static_cast<std::size_t>(i)], sn = s[static_cast<std::size_t>(i) + 1];
      std::copy(x.begin(), x.end(), static_cast<float*>(v.x.data));
      mi::pipeline::StepArgs a;
      a.sigma = si;
      a.x = v.x;
      a.out = v.out;
      const mi::pipeline::StepReport rep = session.step(a);
      EXPECT_EQ(rep.step, i);
      predict(v, x, si, rep.rows, pred);
      if (denoised) {
        // SamplerDriver's euler on a denoised prediction: d = (x - D) / s, x = x + d * fp32(sn - si).
        const float dt = sn - si;
        for (std::size_t k = 0; k < x.size(); ++k) {
          const float e = x[k] - pred[k];
          const float d = e / si;
          const float t = d * dt;
          x[k] = x[k] + t;
        }
      } else {
        // The official euler: h = fp32(double(sn) - double(si)), x = x + h * v.
        const float h = static_cast<float>(static_cast<double>(sn) - static_cast<double>(si));
        for (std::size_t k = 0; k < x.size(); ++k) {
          const float t = h * pred[k];
          x[k] = x[k] + t;
        }
      }
    }
    const std::optional<mi::pipeline::RunResult> done = session.end_run();
    EXPECT_TRUE(done.has_value());
    return x;
  };
  const auto differing = [](const std::vector<float>& a, const float* b, std::size_t n) {
    std::size_t d = 0;
    for (std::size_t k = 0; k < n; ++k) d += a[k] != b[k];
    return d;
  };

  RunSpec spec;
  spec.width = q.width;
  spec.height = q.height;
  spec.steps = q.steps;
  spec.rows = {{conds[0], {}, 0}, {conds[1], {22}, 0}};
  const std::size_t n = ref7.latents.size();
  const Predict host_cfg = [&](const mi::pipeline::StagingViews& v, const std::vector<float>& x, float, std::uint32_t,
                               std::vector<float>& pred) {
    const std::size_t L = x.size() / n;
    pred.resize(x.size());
    for (std::size_t l = 0; l < L; ++l) {
      cfg(static_cast<const float*>(v.out[2 * l].data), static_cast<const float*>(v.out[2 * l + 1].data),
          pred.data() + l * n, n);
    }
  };
  {
    SCOPED_TRACE("HOST rows, the host's CFG");
    spec.guidance = GuidanceMode::kHost;
    spec.output = StepOutput::kRows;
    const std::vector<float> x = sample(spec, {q.seed}, false, host_cfg);
    EXPECT_EQ(differing(x, ref7.latents.data(), n), 0u) << "of " << n << " latent values";
  }
  {
    SCOPED_TRACE("MOSS rows, the host's combine (CFG on every step: no windows)");
    spec.guidance = GuidanceMode::kMoss;
    spec.output = StepOutput::kRows;
    const std::vector<float> x = sample(spec, {q.seed}, false, [&](const mi::pipeline::StagingViews& v,
                                                                  const std::vector<float>& xx, float, std::uint32_t rows,
                                                                  std::vector<float>& pred) {
      ASSERT_EQ(rows, 3u);
      pred.resize(xx.size());
      cfg(static_cast<const float*>(v.out[0].data), static_cast<const float*>(v.out[1].data), pred.data(), n);
    });
    EXPECT_EQ(differing(x, ref7.latents.data(), n), 0u) << "of " << n << " latent values";
  }
  {
    SCOPED_TRACE("MOSS GUIDED_DENOISED, euler on simple");
    RunSpec ds = spec;
    ds.guidance = GuidanceMode::kMoss;
    ds.output = StepOutput::kGuidedDenoised;
    ds.scheduler = mi::pipeline::SchedulerKind::kSimple;
    const std::vector<float> x = sample(ds, {q.seed}, true, [&](const mi::pipeline::StagingViews& v,
                                                               const std::vector<float>& xx, float, std::uint32_t,
                                                               std::vector<float>& pred) {
      const auto* o = static_cast<const float*>(v.out[0].data);
      pred.assign(o, o + xx.size());
    });
    EXPECT_EQ(differing(x, ref_simple.latents.data(), n), 0u) << "of " << n << " latent values";
  }
  {
    SCOPED_TRACE("HOST latent batch of 2");
    RunSpec bs = spec;
    bs.guidance = GuidanceMode::kHost;
    bs.output = StepOutput::kRows;
    bs.rows = {{conds[0], {}, 0}, {conds[1], {22}, 0}, {conds[0], {}, 1}, {conds[1], {22}, 1}};
    const std::vector<float> x = sample(bs, {q.seed, q8.seed}, false, host_cfg);
    ASSERT_EQ(x.size(), 2 * n);
    EXPECT_EQ(differing(x, ref7.latents.data(), n), 0u) << "image 0 of the batch";
    std::vector<float> second(x.begin() + static_cast<std::ptrdiff_t>(n), x.end());
    EXPECT_EQ(differing(second, ref8.latents.data(), n), 0u) << "image 1 of the batch";
  }
  EXPECT_TRUE(session.close());
}

// A latent batch whose latents' forwards differ in shape (prompts of different lengths) with the CUDA-graph block path:
// the forwards alternate between two step shapes, and both replay their graphs (the DiT's graph window is the run's
// latent count); each image still equals mi_generate's final latents.
TEST_F(GpuSession, LatentForwardsOfDifferentShapesReplayGraphs) {
  mi::pipeline::PipelineOptions po = pipeline_options();
  po.block_path = mi::moss_dit::BlockPath::kFusedGraphs;
  mi::pipeline::Pipeline pipe(po);
  mi::runtime::JobGate gate;
  mi::pipeline::Request qa = moss_request();
  qa.decode = false;
  qa.prompt = "an old lighthouse on a basalt cliff at golden hour, waves breaking into white spray, gulls circling";
  mi::pipeline::Request qb = qa;
  qb.prompt = "a red apple";
  qb.seed = 8;
  pipe.generate(qa);  // loads the models and fills the embedding cache
  const mi::pipeline::Result ra = pipe.generate(qa), rb = pipe.generate(qb);
  ASSERT_FALSE(ra.latents.empty());
  SessionOptions so;
  so.plan_request = qa;
  so.max_rows = 4;
  so.max_latents = 2;
  Session session(pipe, gate, so);
  const std::vector<std::string> prompts = {qa.prompt, qb.prompt, qa.negative_prompt};
  const auto conds = session.encode_texts(prompts, qa.max_sequence_length);
  ASSERT_NE(conds[0]->length(), conds[1]->length());
  RunSpec spec;
  spec.width = qa.width;
  spec.height = qa.height;
  spec.steps = qa.steps;
  spec.rows = {{conds[0], {}, 0}, {conds[2], {22}, 0}, {conds[1], {}, 1}, {conds[2], {22}, 1}};
  session.begin_run(spec);
  const mi::pipeline::RunInfo info = session.run_info();
  const mi::pipeline::StagingViews v = session.staging();
  const std::vector<float> s = session.run_sigmas();
  const std::size_t n = ra.latents.size();
  const float g = static_cast<float>(qa.cfg);
  std::vector<float> x;
  for (const std::int64_t seed : {qa.seed, qb.seed}) {
    const std::vector<float> nz =
        mi::pipeline::initial_noise(qa.noise_layout, static_cast<std::uint64_t>(seed), 128, info.latent_height, info.latent_width);
    for (const float e : nz) {
      const float p = s[0] * e;
      const float z = (1.0f - s[0]) * 0.0f;
      x.push_back(p + z);
    }
  }
  for (int i = 0; i < info.steps; ++i) {
    const float si = s[static_cast<std::size_t>(i)], sn = s[static_cast<std::size_t>(i) + 1];
    std::copy(x.begin(), x.end(), static_cast<float*>(v.x.data));
    mi::pipeline::StepArgs a;
    a.sigma = si;
    a.x = v.x;
    a.out = v.out;
    session.step(a);
    const float h = static_cast<float>(static_cast<double>(sn) - static_cast<double>(si));
    for (std::size_t l = 0; l < 2; ++l) {
      const auto* c = static_cast<const float*>(v.out[2 * l].data);
      const auto* u = static_cast<const float*>(v.out[2 * l + 1].data);
      for (std::size_t k = 0; k < n; ++k) {
        const float d = c[k] - u[k];
        const float t = g * d;
        const float vel = u[k] + t;
        const float m = h * vel;
        x[l * n + k] = x[l * n + k] + m;
      }
    }
  }
  const std::optional<mi::pipeline::RunResult> done = session.end_run();
  ASSERT_TRUE(done.has_value());
  EXPECT_GT(done->graph_replays, 0u) << "the latents' alternating step shapes never replayed their graphs";
  std::size_t da = 0, db = 0;
  for (std::size_t k = 0; k < n; ++k) {
    da += x[k] != ra.latents[k];
    db += x[n + k] != rb.latents[k];
  }
  EXPECT_EQ(da, 0u) << "image 0 differs from mi_generate's final latents";
  EXPECT_EQ(db, 0u) << "image 1 differs from mi_generate's final latents";
  EXPECT_TRUE(session.close());
}

// Refine dedup by handle (design §5.4): a HOST run whose two rows share one conditioning handle and skip set (one refined
// text, one step-cache slot) equals the run with two distinct handles of equal content (two refined texts, two slots),
// in bits and cached steps, with the step cache on. Device tensors on the legacy stream (the event handoff).
TEST_F(GpuSession, SharedHandleEqualsDistinctHandlesWithTheStepCache) {
  mi::pipeline::Pipeline pipe(pipeline_options());
  mi::runtime::JobGate gate;
  const mi::pipeline::Request q = moss_request();
  SessionOptions so;
  so.plan_request = q;
  Session session(pipe, gate, so);
  const std::vector<std::string> prompts = {q.prompt};
  const std::shared_ptr<const Conditioning> c = session.encode_texts(prompts, q.max_sequence_length)[0];
  const auto twin = std::make_shared<Conditioning>(*c);  // another handle, the same embedding
  const std::int64_t h = q.height / 16, w = q.width / 16, n = 128 * h * w;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  mi::DeviceBuffer xd(bytes), o0(bytes), o1(bytes);
  const auto view = [&](mi::DeviceBuffer& b) { return mi::Tensor::view(b.get(), mi::DType::kF32, mi::Device::kCUDA, {128, h, w}); };
  const auto sample = [&](const std::shared_ptr<const Conditioning>& second, std::vector<int>& cached) {
    RunSpec spec;
    spec.width = q.width;
    spec.height = q.height;
    spec.steps = 8;
    spec.rows = {{c, {}, 0}, {second, {}, 0}};
    spec.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;
    spec.step_cache_threshold = 0.5;
    session.begin_run(spec);
    std::vector<float> x = mi::pipeline::initial_noise(q.noise_layout, 11, 128, h, w), v0(x.size()), v1(x.size());
    const std::vector<float> s = session.run_sigmas();
    for (int i = 0; i + 1 < static_cast<int>(s.size()); ++i) {
      MI_CUDA_CHECK(cudaMemcpy(xd.get(), x.data(), bytes, cudaMemcpyHostToDevice));
      mi::pipeline::StepArgs a;
      a.sigma = s[static_cast<std::size_t>(i)];
      a.x = view(xd);
      a.out[0] = view(o0);
      a.out[1] = view(o1);
      if (session.step(a).cached) cached.push_back(i);
      MI_CUDA_CHECK(cudaMemcpy(v0.data(), o0.get(), bytes, cudaMemcpyDeviceToHost));
      MI_CUDA_CHECK(cudaMemcpy(v1.data(), o1.get(), bytes, cudaMemcpyDeviceToHost));
      const float dt = s[static_cast<std::size_t>(i) + 1] - s[static_cast<std::size_t>(i)];
      EXPECT_TRUE(v0 == v1) << "two rows of equal inputs differ at step " << i;
      for (std::size_t k = 0; k < x.size(); ++k) {
        const float t = dt * v0[k];
        x[k] = x[k] + t;
      }
    }
    session.end_run();
    return x;
  };
  std::vector<int> cached_shared, cached_distinct;
  const std::vector<float> shared = sample(c, cached_shared);
  const std::vector<float> distinct = sample(twin, cached_distinct);
  EXPECT_EQ(cached_shared, cached_distinct);
  EXPECT_TRUE(shared == distinct);
}

}  // namespace
