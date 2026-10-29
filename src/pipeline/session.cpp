#include "pipeline/session.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <utility>

#include "core/cuda_check.h"
#include "models/control/control.h"
#include "models/qwen35_text/text_encoder.h"
#include "pipeline/accel/accel.h"
#include "pipeline/pipeline_impl.h"
#include "pipeline/row_select.h"
#include "pipeline/run_schedule.h"
#include "pipeline/sampler.h"
#include "planner/host_memory.h"
#include "runtime/exchange.h"
#include "tokenizer/prompt_tokens.h"
#include "tokenizer/qwen_bpe.h"

namespace mi::pipeline {
namespace {

using runtime::JobGate;

// What a session's plans size the denoise phase for: a run's shape (the last run's for encode_texts and the VAE calls,
// plan_request's before the first run). Pipeline::run's PlanCall of the equivalent request is built from it.
struct Shape {
  int height = 1024, width = 1024;
  int latents = 1;
  std::vector<std::int64_t> texts;  // the refined texts' tokens (the distinct conditionings of the evaluated rows)
  std::vector<std::int64_t> rows;   // the largest call's rows (each row's text tokens), in row order
  std::int64_t host_tokens = 0;     // the text-encoder states the equivalent request holds on the host
  int steps = 1;
  bool crops = false;
};

int resident_blocks(const planner::DitPlan& d) {
  return d.residency == planner::DitResidency::kStreamed ? d.resident_blocks : moss_dit::kLayers;
}

// MOSS guidance of a run (mi_generate's Guidance); HOST runs evaluate their rows as given (cfg 1: no guidance).
Guidance run_guidance(const RunSpec& spec) {
  Guidance g;
  g.cfg = 1.0;
  if (spec.guidance != GuidanceMode::kMoss) return g;
  if (spec.rows.size() == 2) {
    g.cfg = spec.cfg;
    g.uncond_skip = skip_set(spec.rows[1].skip_layers);
  }
  g.slg_layers = skip_set(spec.slg_layers);
  g.slg_scale = spec.slg_scale;
  g.slg_start = spec.slg_start;
  g.slg_stop = spec.slg_stop;
  g.cfg_truncation = spec.cfg_truncation;
  return g;
}

bool slg_enabled(const Guidance& g) { return g.slg_layers.any() && g.slg_scale > 0; }

// The run's row slots by row bit (row_select.h: MOSS 0 cond, 1 uncond, 2 SLG; HOST row k) that a call may evaluate,
// with the spec row each reads its conditioning from.
std::vector<std::pair<int, int>> evaluated_rows(const RunSpec& spec, const Guidance& g, bool latent0_only) {
  std::vector<std::pair<int, int>> out;  // (row bit, spec row)
  if (spec.guidance == GuidanceMode::kMoss) {
    out.emplace_back(0, 0);
    if (g.do_cfg()) out.emplace_back(1, 1);
    if (slg_enabled(g)) out.emplace_back(2, 0);
    return out;
  }
  for (std::size_t k = 0; k < spec.rows.size(); ++k) {
    if (!latent0_only || spec.rows[k].latent == 0) out.emplace_back(static_cast<int>(k), static_cast<int>(k));
  }
  return out;
}

int latents_of(const RunSpec& spec) {
  int l = 0;
  for (const RunRow& r : spec.rows) l = std::max(l, r.latent + 1);
  return l;
}

Shape run_shape(const RunSpec& spec, const Guidance& g, int steps, bool latent0_only) {
  Shape sh;
  sh.height = spec.height;
  sh.width = spec.width;
  sh.latents = latent0_only ? 1 : latents_of(spec);
  sh.steps = steps;
  sh.crops = spec.crops;
  std::vector<const Conditioning*> seen;
  for (const auto& [bit, k] : evaluated_rows(spec, g, latent0_only)) {
    const Conditioning& c = *spec.rows[static_cast<std::size_t>(k)].conditioning;
    sh.rows.push_back(c.length());
    if (bit == 2 && spec.guidance == GuidanceMode::kMoss) continue;  // the SLG row reads the cond row's text
    sh.host_tokens += c.length();
    // One refined text per distinct conditioning handle (pointer identity; equal contents in two handles count twice).
    if (std::find(seen.begin(), seen.end(), &c) != seen.end()) continue;
    seen.push_back(&c);
    sh.texts.push_back(c.length());
  }
  return sh;
}

// The session request of a run: the equivalent mi_generate request (sampler euler, decode on), validated as one.
Request run_request(const SessionOptions& o, const RunSpec& spec, const Guidance& g) {
  Request q;
  q.model = o.plan_request.model;
  q.prompt = "component";  // tokenized by nothing: the rows carry their conditionings
  q.height = spec.height;
  q.width = spec.width;
  q.steps = spec.steps;
  q.cfg = g.cfg;
  q.shift = spec.shift;
  q.flow_multiplier = spec.flow_multiplier;
  q.scheduler = spec.scheduler;
  q.sampler = SamplerKind::kEuler;
  q.uncond_skip_layers.clear();
  if (spec.guidance == GuidanceMode::kMoss && spec.rows.size() == 2) q.uncond_skip_layers = spec.rows[1].skip_layers;
  q.slg_layers = spec.slg_layers;
  q.slg_scale = spec.slg_scale;
  q.slg_start = spec.slg_start;
  q.slg_stop = spec.slg_stop;
  q.loras = spec.loras;
  q.control.adapter = o.control_adapter;
  q.control.lora_strength = spec.control_lora_strength;
  q.control.images = spec.control_images;
  q.resize = spec.control_resize;
  q.accel.cfg_truncation = spec.cfg_truncation;
  q.accel.step_cache = spec.step_cache;
  if (spec.step_cache != accel::StepCacheKind::kOff) q.accel.step_cache_threshold = spec.step_cache_threshold;
  q.decode = true;
  return q;
}

// The schedule a run's positions index: the host's sigmas, else the engine's for the scheduler fields.
std::vector<float> run_sigmas_of(const RunSpec& spec) {
  if (!spec.sigmas.empty()) return spec.sigmas;
  return engine_sigmas(spec.scheduler, spec.steps, FlowSampling{spec.shift, spec.flow_multiplier}, spec.denoise_strength);
}

void expect_tensor(const Tensor& t, DType dtype, std::initializer_list<std::int64_t> shape, const std::string& what) {
  bool ok = t.data != nullptr && t.dtype == dtype && t.rank == static_cast<int>(shape.size()) && t.is_contiguous();
  int i = 0;
  for (const std::int64_t d : shape) ok = ok && t.dim(i++) == d;
  if (ok) return;
  std::string want = std::string(dtype == DType::kU8 ? "u8" : "f32") + "[";
  i = 0;
  for (const std::int64_t d : shape) want += (i++ ? ", " : "") + std::to_string(d);
  throw std::invalid_argument(what + ": expected " + want + "], got " + (t.data == nullptr ? std::string("none") : t.describe()));
}

bool overlaps(const Tensor& a, const Tensor& b) {
  const auto* pa = static_cast<const std::byte*>(a.data);
  const auto* pb = static_cast<const std::byte*>(b.data);
  return pa < pb + b.nbytes() && pb < pa + a.nbytes();
}

void sync_quietly(cudaStream_t stream) { (void)cudaStreamSynchronize(stream); }

}  // namespace

struct Session::Run {
  Result result;
  RunSpec spec;  // its rows keep their conditionings alive
  Guidance guidance;
  bool moss = false;
  std::optional<RunSchedule> schedule;
  bool schedule_matches = false;
  std::string schedule_note;
  int latents = 1;
  std::int64_t h = 0, w = 0;
  std::uint32_t row_bits = 0;  // the rows a call may evaluate
  int outputs = 0;
  std::array<int, kMaxSessionRows> slot{};  // output k's staging slot (-1: a row the run lacks)
  std::array<moss_dit::Row, kMaxSessionRows> rows{};  // by row bit (MOSS: the uncond row's skip set applied per call)
  std::array<Tensor, kMaxSessionRows> velocities{};   // by row bit, on the run's stack
  Tensor guided;                                       // a guided output staged on the device (host tensors)
  std::optional<runtime::StackScope> stack;
  Tensor x;                                            // the latents staged on the device (host tensors)
  const control::ControlFile* ctl = nullptr;
  Tensor ctl_extra;                                    // [L, G * 128, h, w]: latent 0 the run's control latents
  std::optional<control::PassInputs> ctl_in;
  bool ctl_dirty = false;  // ctl_extra holds a crop's or a call's control: rewritten before the run's are used again
  std::optional<accel::FirstBlockCache> cache;
  runtime::StagingLayout layout;
  runtime::StagingBlock block;
  std::unique_ptr<runtime::DevicePointerCheck> check;
  std::vector<char> stepped;  // steps with a completed call
  std::chrono::steady_clock::time_point started;
};

struct Session::State {
  Pipeline::Impl& m;
  JobGate& gate;
  SessionOptions options;
  Request request;  // the session request
  std::optional<Shape> shape;
  runtime::PinnedStaging staging;
  std::unique_ptr<runtime::StreamHandoff> handoff;
  std::unique_ptr<Run> run;

  State(Pipeline::Impl& impl, JobGate& g, SessionOptions o) : m(impl), gate(g), options(std::move(o)) {}

  // A call's report and the pipeline's per-call memory counters (Impl::mem) pointing into it while the call runs.
  struct CallScope {
    Pipeline::Impl& m;
    CallScope(Pipeline::Impl& impl, Result& r) : m(impl) { m.mem = &r.memory; }
    ~CallScope() { m.mem = nullptr; }
  };
  void start_report(Result& r) const {
    r.tier.tier = m.options.tier;
    r.memory.free_at_start = m.free_at_start;
    r.memory.host_private_start = planner::process_host_memory().private_bytes;
    r.memory.host_private_peak = r.memory.host_private_start;
  }
  void finish_report(Result& r) const {
    r.memory.host_copy_bytes = m.te_host_bytes() + m.dit_host_bytes();
    r.memory.merge_memo_bytes = m.dit ? m.dit->memo_bytes() : 0;
    m.sample_memory();
  }
  bool cancelled() const { return gate.cancel_flag()->load(); }
  // Before a phase: the callback, then the cancel flag.
  void enter(Phase phase) const {
    if ((options.on_phase && !options.on_phase(phase)) || cancelled()) throw SessionCancelled("the call was cancelled");
  }

  // plan_request's shape: its prompt (a short one when blank) and negative prompt tokenized, its rows and steps.
  Shape request_shape(const Bundle& b) const {
    const Request& q = options.plan_request;
    Shape sh;
    sh.height = q.height;
    sh.width = q.width;
    tok::PromptTokenReport report;
    const std::string prompt = tok::python_strip(q.prompt).empty() ? std::string("preload") : q.prompt;
    const auto len = static_cast<std::size_t>(q.max_sequence_length);
    const std::int64_t sc = static_cast<std::int64_t>(tok::encode_moss_prompt_report(b.tokenizer, prompt, len, report).size());
    Guidance g;
    g.cfg = q.cfg;
    g.slg_layers = skip_set(q.slg_layers);
    g.slg_scale = q.slg_scale;
    const std::int64_t su =
        g.do_cfg() ? static_cast<std::int64_t>(tok::encode_moss_prompt_report(b.tokenizer, q.negative_prompt, len, report).size()) : 0;
    sh.texts = {sc};
    if (g.do_cfg()) sh.texts.push_back(su);
    sh.rows = denoise_row_lengths(sc, su, g);
    sh.host_tokens = sc + su;
    sh.steps = static_cast<int>(schedule_steps(q));
    return sh;
  }

  // The denoise phase of `sh` at H x W (denoise_stack_bytes, as Pipeline::run's denoise_need_of for sampler euler).
  std::size_t denoise_need(const Shape& sh, int H, int W, const Pipeline::Impl::CallDit& d, const control::ControlFile* ctl,
                           bool step_cache) const {
    const std::int64_t lh = H / 16, lw = W / 16;
    DenoiseStack ds;
    ds.height = H;
    ds.width = W;
    ds.latents = sh.latents;
    ds.texts = sh.texts;
    ds.rows = sh.rows;
    ds.loop_bytes = sh.rows.size() * latent_bytes_of(H, W) +
                    SamplerDriver::workspace_bytes(SamplerKind::kEuler, moss_dit::kLatentChannels * lh * lw, kMaxDenoiseSteps,
                                                   moss_dit::kLatentChannels) +
                    (step_cache ? accel::FirstBlockCache::workspace_bytes(lh, lw, static_cast<int>(sh.rows.size())) : 0);
    ds.control_groups = ctl != nullptr ? ctl->groups() : 0;
    ds.crops = sh.crops;
    return denoise_stack_bytes(ds, d.block_path, d.config, d.route);
  }

  // The session request with shape `sh` at H x W planned as Pipeline::run plans it (plan_call; te_lengths empty: an
  // embedding-cache hit), into r.plan. `encoder`: the VAE encoder the call uses.
  Pipeline::Impl::Planned plan(const Pipeline::Impl::CallModels& models, const Pipeline::Impl::CallDit& d, const Shape& sh,
                               int H, int W, const EncoderSource* encoder, std::vector<std::int64_t> te_lengths,
                               std::size_t& denoise, Result& r) const {
    const bool ctl_encode = models.ctl != nullptr && !request.control.images.empty();
    denoise = denoise_need(sh, H, W, d, models.ctl, request.accel.step_cache_on());
    Pipeline::Impl::PlanCall pc;
    pc.height = H;
    pc.width = W;
    pc.text_lengths = sh.texts;
    pc.te_lengths = std::move(te_lengths);
    pc.steps = sh.steps;
    pc.rows_per_step = static_cast<int>(sh.rows.size());
    pc.need_dit = true;
    pc.encoder = encoder;
    pc.decode_height = H;
    pc.decode_width = W;
    pc.denoise = std::max(denoise, encoder != nullptr ? encode_need_of(H, W) : 0);
    const std::size_t lat = latent_bytes_of(H, W), img = image_bytes_of(H, W);
    pc.host_buffers = static_cast<std::size_t>(sh.host_tokens) * qwen35_text::kHidden * 2 + 2 * lat + img;
    if (ctl_encode) pc.host_buffers += request.control.images.size() * (lat + img);
    return m.plan_call(models, pc, d.block_path, r);
  }

  // After the call's last plan: the explanation's model and control lines, the embedding cache's cap and the merge memo
  // (as Pipeline::run), then the plan's arena.
  void apply_plan(const Pipeline::Impl::CallModels& models, const Pipeline::Impl::Planned& planned, Result& r) {
    m.explain_models(*models.src, r);
    if (models.ctl != nullptr) Pipeline::Impl::explain_control(request.control, r);
    if (m.cache) m.cache->set_max_bytes(r.plan.cache.host_bytes);
    if (m.dit && m.dit->memo_bytes() > 0 && merge_memo_unaffordable(r.plan, m.options.host_ram_budget)) m.dit->drop_merge_memo();
    m.setup_arena(r.plan, planned.kept, true, planned.drop_resident, r);
  }

  // setup_models, then switch_merges for a loaded DiT of the call's source (as Pipeline::run).
  Pipeline::Impl::CallModels models_for(const Request& q, Result& r) {
    check_user_paths(m.options, q);
    Pipeline::Impl::CallModels models = m.setup_models(q, r);
    if (m.dit && m.dit_src == models.src && m.dit_merge_key != models.merge_key) m.switch_merges(models.merge_key, r.timings);
    return models;
  }

  // The DiT's per-run settings cleared (as Pipeline::run's CancelScope): its stack-held inputs and the gate's flag.
  void clear_dit() const {
    if (!m.dit) return;
    m.dit->set_cancel(nullptr);
    m.dit->set_request_route(std::nullopt);
    m.dit->set_input_extension(nullptr);
    m.dit->set_step_cache(nullptr);
    m.dit->set_graphs_paused(false);
  }

  // The most pinned host bytes the staging may hold: the planner's pin margin of the available host memory, and the
  // host budget when one is set.
  std::size_t staging_budget() const {
    const std::size_t available = planner::host_available_bytes();
    const std::size_t margin = planner::host_pin_margin(planner::host_total_bytes());
    std::size_t budget = available == 0 ? std::numeric_limits<std::size_t>::max()
                                        : staging.bytes() + (available > margin ? available - margin : 0);
    if (m.options.host_ram_budget != 0) {
      const std::size_t used = planner::process_host_memory().private_bytes;
      budget = std::min(budget, staging.bytes() + (m.options.host_ram_budget > used ? m.options.host_ram_budget - used : 0));
    }
    return budget;
  }

  RunResult finish_run();
};

Session::Session(Pipeline& pipeline, JobGate& gate, SessionOptions options) {
  if (options.max_rows < 1 || options.max_rows > kMaxSessionRows) {
    throw std::invalid_argument("max_rows must be 1 to " + std::to_string(kMaxSessionRows) + ".");
  }
  if (options.max_latents < 1 || options.max_latents > options.max_rows) {
    throw std::invalid_argument("max_latents must be 1 to max_rows (" + std::to_string(options.max_rows) + ").");
  }
  Request q = options.plan_request;
  q.control.adapter = options.control_adapter;
  if (q.control.adapter.empty()) q.control.images.clear();
  if (tok::python_strip(q.prompt).empty()) q.prompt = "preload";
  check_user_paths(pipeline.impl_->options, q);
  pipeline::validate(q);
  if (q.accel.step_cache_on() && !q.accel.step_cache_threshold) {
    throw std::invalid_argument("The step cache needs an explicit threshold in (0, 1] (step_cache_threshold).");
  }
  JobGate::Job job = gate.begin(JobGate::Kind::kEngine);
  if (gate.session_open()) throw JobGate::Busy("a session is open on this engine");
  auto s = std::make_unique<State>(*pipeline.impl_, gate, std::move(options));
  s->request = std::move(q);
  s->handoff = std::make_unique<runtime::StreamHandoff>();
  s_ = std::move(s);
  gate.open_session();
}

Session::~Session() {
  try {
    close();
  } catch (...) {
  }
}

bool Session::close() {
  State& s = *s_;
  return s.gate.end_session([&] { s.finish_run(); },
                            [&] {
                              s.staging.release();
                              s.handoff.reset();
                            });
}

void Session::validate(const RunSpec& spec, const SessionOptions& options) {
  const bool moss = spec.guidance == GuidanceMode::kMoss;
  const auto n = static_cast<int>(spec.rows.size());
  if (n < 1 || n > options.max_rows) {
    throw std::invalid_argument("A run takes 1 to max_rows (" + std::to_string(options.max_rows) + ") rows, not " +
                                std::to_string(n) + ".");
  }
  std::vector<char> used(static_cast<std::size_t>(options.max_latents), 0);
  for (int k = 0; k < n; ++k) {
    const RunRow& row = spec.rows[static_cast<std::size_t>(k)];
    const std::string what = "rows[" + std::to_string(k) + "]";
    if (!row.conditioning || !row.conditioning->embedding) throw std::invalid_argument(what + ": no conditioning.");
    const Embedding& e = *row.conditioning->embedding;
    const std::int64_t tokens = e.hidden == moss_dit::kTextDim ? static_cast<std::int64_t>(e.data.size()) / e.hidden : 0;
    if (tokens < 1 || tokens > 4096 || e.data.size() != static_cast<std::size_t>(tokens) * moss_dit::kTextDim) {
      throw std::invalid_argument(what + ": the conditioning must be [1..4096, " + std::to_string(moss_dit::kTextDim) + "].");
    }
    check_layers(row.skip_layers, "skip_layers");
    if (row.latent < 0 || row.latent >= options.max_latents) {
      throw std::invalid_argument(what + ": latent " + std::to_string(row.latent) + " is outside the session's 0.." +
                                  std::to_string(options.max_latents - 1) + ".");
    }
    used[static_cast<std::size_t>(row.latent)] = 1;
  }
  const int latents = latents_of(spec);
  for (int l = 0; l < latents; ++l) {
    if (!used[static_cast<std::size_t>(l)]) {
      throw std::invalid_argument("The rows' latents must cover 0.." + std::to_string(latents - 1) + ": no row reads latent " +
                                  std::to_string(l) + ".");
    }
  }
  if (moss) {
    if (n > 2 || latents != 1) throw std::invalid_argument("MOSS guidance takes 1 or 2 rows (cond, uncond) on latent 0.");
    if (spec.output != StepOutput::kRows && n == 2 && spec.cfg == 1.0) {
      throw std::invalid_argument("Guided outputs with an uncond row need cfg != 1.");
    }
  } else {
    if (spec.output != StepOutput::kRows) throw std::invalid_argument("Guided outputs need MOSS guidance.");
    if (spec.cfg != 3.0 || !spec.slg_layers.empty() || spec.slg_scale != 0.0 || spec.slg_start != 0.0 ||
        spec.slg_stop != 1.0 || spec.cfg_truncation != 1.0) {
      throw std::invalid_argument("HOST guidance keeps cfg, slg_* and cfg_truncation at their defaults (the host guides).");
    }
  }
  if (options.control_adapter.empty() && !spec.control_images.empty()) {
    throw std::invalid_argument("Control images need the session's control adapter.");
  }
  if (spec.step_cache != accel::StepCacheKind::kOff && !(spec.step_cache_threshold > 0 && spec.step_cache_threshold <= 1)) {
    throw std::invalid_argument("The step cache needs an explicit threshold in (0, 1] (step_cache_threshold).");
  }
  // The rules mi_generate's request shares (sizes, steps, schedule fields, CFG, SLG, control images, accelerators).
  const Guidance g = run_guidance(spec);
  pipeline::validate(run_request(options, spec, g));
  if (spec.sigmas.empty() && !(std::isfinite(spec.denoise_strength) && spec.denoise_strength >= 0 && spec.denoise_strength <= 1)) {
    throw std::invalid_argument("Denoise strength must be between 0 and 1.");
  }
  if (spec.sigmas.empty() && run_sigmas_of(spec).empty()) {
    throw std::invalid_argument("The schedule has no step left (denoise strength 0): the VAE round trip needs no run.");
  }
  if (!spec.sigmas.empty()) RunSchedule(spec.sigmas, spec.exact_sigma);  // its errors name the first bad value
  std::int64_t tokens = 0;
  const std::int64_t video = std::int64_t{spec.height / 16} * (spec.width / 16);
  for (const std::int64_t s : run_shape(spec, g, 1, false).rows) tokens += video + s;
  if (tokens > kMaxCallTokens) {
    throw std::invalid_argument("The run's largest call would carry " + std::to_string(tokens) + " tokens, more than the " +
                                std::to_string(kMaxCallTokens) + " the DiT's fused path validated (MI_MAX_CALL_TOKENS): use " +
                                "fewer rows or latents per run, or a smaller size.");
  }
}

std::vector<std::shared_ptr<const Conditioning>> Session::encode_texts(std::span<const std::string> prompts,
                                                                       int max_sequence_length) {
  State& s = *s_;
  JobGate::Job job = s.gate.begin(JobGate::Kind::kSession);
  if (prompts.empty() || prompts.size() > 8) throw std::invalid_argument("mi_encode_text takes 1 to 8 prompts.");
  if (max_sequence_length < 1 || max_sequence_length > 4096) {
    throw std::invalid_argument("Maximum text length must be between 1 and 4096.");
  }
  Pipeline::Impl& m = s.m;
  Result r;
  s.start_report(r);
  State::CallScope scope(m, r);
  const Pipeline::Impl::CallModels models = s.models_for(s.request, r);
  const TeSource& tes = *models.tes;
  std::vector<std::shared_ptr<Conditioning>> out(prompts.size());
  std::vector<EmbeddingKey> keys(prompts.size());
  std::vector<std::size_t> missing;
  std::vector<std::int64_t> te_lengths;
  std::vector<std::vector<std::int32_t>> todo;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    out[i] = std::make_shared<Conditioning>();
    out[i]->engine_encoded = true;
    std::vector<std::int32_t> seq = tok::encode_moss_prompt_report(models.b->tokenizer, prompts[i],
                                                                   static_cast<std::size_t>(max_sequence_length), out[i]->tokens);
    keys[i] = EmbeddingKey{tes.identity, tes.numerics, max_sequence_length, seq};
    if (m.cache) out[i]->embedding = m.cache->find(keys[i]);
    if (!out[i]->embedding) {
      missing.push_back(i);
      te_lengths.push_back(static_cast<std::int64_t>(seq.size()));
      todo.push_back(std::move(seq));
    }
  }
  if (!missing.empty()) {
    s.enter(Phase::kTextEncoder);
    if (!s.shape) s.shape = s.request_shape(*models.b);
    const Pipeline::Impl::CallDit d = m.setup_dit_call(*models.src, r);
    const bool ctl_encode = models.ctl != nullptr && !s.request.control.images.empty();
    const EncoderSource* enc = ctl_encode ? &m.encoder_source(*models.b, r.timings.index_load_s) : nullptr;
    std::size_t denoise = 0;
    const Pipeline::Impl::Planned planned =
        s.plan(models, d, *s.shape, s.shape->height, s.shape->width, enc, te_lengths, denoise, r);
    s.apply_plan(models, planned, r);
    const planner::ExecutionPlan& plan = r.plan;
    // As Pipeline::run's text-encoder phase: the plan may park the DiT for it.
    if (plan.dit.park_for_text_encoder && (m.dit || m.vae || m.encoder)) {
      r.memory.dit_parked_for_text_encoder = m.dit != nullptr;
      m.reset_arena(m.arena->capacity(), true);
    }
    const auto t0 = Clock::now();
    m.ensure_te(tes, m.te_options(tes, plan.te.residency, plan.te.buffers, plan.te.pinned));
    qwen35_text::EncodeOptions eo;
    eo.cancel = s.gate.cancel_flag();
    std::vector<std::vector<std::uint16_t>> encoded;
    try {
      m.load_te_host(tes, plan.te.buffers, eo.cancel, r);
      encoded = m.te->encode_to_host(todo, eo);
    } catch (const qwen35_text::Cancelled&) {
      throw SessionCancelled("the text encoder was cancelled");
    }
    r.timings.text_encode_s = since(t0);
    const qwen35_text::EncodeStats& st = m.te->last_stats();
    if (m.te->options().source == qwen35_text::WeightSource::kFile && m.learner && !m.options.direct_io) {
      m.learner->observe_te_stream(planner::weights_cached_from_stream(st.weight_bytes, st.weight_load_s, st.weight_wait_s, r.hardware));
    }
    for (std::size_t k = 0; k < missing.size(); ++k) {
      auto e = std::make_shared<Embedding>();
      e->hidden = qwen35_text::kHidden;
      e->data = std::move(encoded[k]);
      // The second prompt (the recipe's negative) is kept, as mi_generate keeps its negative prompt.
      if (m.cache) m.cache->insert(keys[missing[k]], *e, missing[k] == 1);
      out[missing[k]]->embedding = std::move(e);
    }
    s.finish_report(r);
  }
  return {out.begin(), out.end()};
}

void Session::begin_run(const RunSpec& spec) {
  State& s = *s_;
  JobGate::Job job = s.gate.begin(JobGate::Kind::kSession);
  validate(spec, s.options);
  Pipeline::Impl& m = s.m;
  auto run = std::make_unique<Run>();
  Run& R = *run;
  Result& r = R.result;
  R.started = Clock::now();
  s.start_report(r);
  State::CallScope scope(m, r);
  // The DiT's per-run settings are cleared unless the run opens.
  struct Clear {
    const State& s;
    bool keep = false;
    ~Clear() {
      if (!keep) s.clear_dit();
    }
  } clear{s};

  R.spec = spec;
  R.moss = spec.guidance == GuidanceMode::kMoss;
  R.guidance = run_guidance(spec);
  const Guidance& g = R.guidance;
  R.schedule.emplace(run_sigmas_of(spec), spec.exact_sigma);
  const std::vector<float> engine =
      engine_sigmas(spec.scheduler, spec.steps, FlowSampling{spec.shift, spec.flow_multiplier}, spec.denoise_strength);
  R.schedule_matches = first_schedule_difference(R.schedule->sigmas(), engine) == kSchedulesEqual;
  R.schedule_note = schedule_note(R.schedule->sigmas(), engine, spec.scheduler, spec.steps,
                                  FlowSampling{spec.shift, spec.flow_multiplier}, spec.denoise_strength);
  const int n = R.schedule->steps();
  const int H = spec.height, W = spec.width;
  R.latents = latents_of(spec);
  R.h = H / 16;
  R.w = W / 16;
  const Request req = run_request(s.options, spec, g);
  const Shape sh = run_shape(spec, g, n, false);

  // Models, the DiT's call setup and the plan, as Pipeline::run plans the equivalent request.
  const Pipeline::Impl::CallModels models = s.models_for(req, r);
  const Request previous = std::exchange(s.request, req);  // the plan reads the run's control images
  struct Restore {
    State& s;
    const Request& previous;
    bool keep = false;
    ~Restore() {
      if (!keep) s.request = previous;
    }
  } restore{s, previous};
  const Pipeline::Impl::CallDit d = m.setup_dit_call(*models.src, r);
  const bool ctl_encode = models.ctl != nullptr && !spec.control_images.empty();
  const EncoderSource* enc = ctl_encode ? &m.encoder_source(*models.b, r.timings.index_load_s) : nullptr;
  if (enc != nullptr) r.vae_encoder_source = util::path_utf8(enc->file);
  std::size_t denoise = 0;
  // The batch rule (a latent batch never makes the DiT stream more than its latent-0 rows alone) and the crop rule
  // (MI_RUN_CROPS's bound never does either): each plans its reference first, then the run's own plan decides.
  int batch_ref = -1, crop_ref = -1;
  if (R.latents > 1) {
    s.plan(models, d, run_shape(spec, g, n, true), H, W, enc, {}, denoise, r);
    batch_ref = resident_blocks(r.plan.dit);
  }
  if (spec.crops) {
    Shape plain = sh;
    plain.crops = false;
    s.plan(models, d, plain, H, W, enc, {}, denoise, r);
    crop_ref = resident_blocks(r.plan.dit);
  }
  const Pipeline::Impl::Planned planned = s.plan(models, d, sh, H, W, enc, {}, denoise, r);
  const int blocks = resident_blocks(r.plan.dit);
  if (blocks < batch_ref) {
    throw std::invalid_argument("A latent batch of " + std::to_string(R.latents) + " would keep " + std::to_string(blocks) +
                                " DiT blocks resident, fewer than the " + std::to_string(batch_ref) +
                                " of its latent-0 rows alone: run one latent per run instead.");
  }
  if (blocks < crop_ref) {
    throw std::invalid_argument("MI_RUN_CROPS's crop workspace would keep " + std::to_string(blocks) +
                                " DiT blocks resident, fewer than the " + std::to_string(crop_ref) + " without it.");
  }
  s.apply_plan(models, planned, r);

  // Control images (encoded before the DiT load, as Pipeline::run), then the DiT.
  std::vector<std::vector<float>> ctl_lat;
  if (ctl_encode) {
    s.enter(Phase::kDecode);
    ctl_lat = m.encode_controls(*enc, *models.ctl, req, H, W, r);
  }
  s.enter(Phase::kDitLoad);
  const moss_dit::Residency residency = m.dit_residency(r.plan.dit);
  if (!m.prepare_dit(*models.src, residency, denoise, d, models.merge_key, s.gate.cancel_flag(), {}, r)) {
    throw SessionCancelled("the DiT load was cancelled");
  }
  const cudaStream_t stream = m.streams.compute();
  runtime::DeviceArena& arena = *m.arena;

  // The run's stack: the refined texts (each distinct conditioning once), the latents, the control latents, the step
  // cache, the rows' velocities and a guided output; each call's forward on top.
  R.stack.emplace(arena);
  std::vector<std::shared_ptr<const Embedding>> texts;
  std::vector<Tensor> refined;
  std::map<const Conditioning*, Tensor> refined_of;
  for (const auto& [bit, k] : evaluated_rows(spec, g, false)) {
    const Conditioning& c = *spec.rows[static_cast<std::size_t>(k)].conditioning;
    if (refined_of.count(&c) != 0) continue;
    const Tensor t = arena.alloc_tensor(DType::kBF16, {c.length(), moss_dit::kDim});
    refined_of.emplace(&c, t);
    texts.push_back(c.embedding);
    refined.push_back(t);
  }
  m.refine_texts(texts, refined, r.timings);
  if (s.cancelled()) throw SessionCancelled("the run was cancelled");
  for (const auto& [bit, k] : evaluated_rows(spec, g, false)) {
    const RunRow& row = spec.rows[static_cast<std::size_t>(k)];
    moss_dit::Row& dr = R.rows[static_cast<std::size_t>(bit)];
    dr.text = refined_of.at(row.conditioning.get());
    dr.latent = row.latent;
    dr.skip = R.moss ? (bit == 2 ? g.slg_layers : bit == 1 ? moss_dit::SkipSet{} : skip_set(row.skip_layers)) : skip_set(row.skip_layers);
    R.row_bits |= 1u << bit;
  }
  R.outputs = spec.output == StepOutput::kRows ? std::bit_width(R.row_bits) : 1;
  const std::int64_t lat = moss_dit::kLatentChannels;
  R.x = R.latents == 1 ? arena.alloc_tensor(DType::kF32, {lat, R.h, R.w})
                       : arena.alloc_tensor(DType::kF32, {R.latents, lat, R.h, R.w});
  R.ctl = models.ctl;
  if (R.ctl != nullptr) {
    const std::int64_t c = R.ctl->groups() * control::kGroupChannels;
    R.ctl_extra = R.latents == 1 ? arena.alloc_tensor(DType::kF32, {c, R.h, R.w})
                                 : arena.alloc_tensor(DType::kF32, {R.latents, c, R.h, R.w});
    if (R.latents > 1) MI_CUDA_CHECK(cudaMemsetAsync(R.ctl_extra.data, 0, R.ctl_extra.nbytes(), stream));
    R.ctl_in.emplace(R.ctl->groups(), R.h, R.w);
    for (const Request::ControlImage& ci : spec.control_images) {
      const int gi = R.ctl->kind_index(ci.kind);
      R.ctl_in->set(gi, std::move(ctl_lat[static_cast<std::size_t>(gi)]), ci.start, ci.end);
    }
    R.ctl_in->apply(Tensor::view(R.ctl_extra.data, DType::kF32, Device::kCUDA, {c, R.h, R.w}), 0, n, stream);
    m.control_ext.extra = R.ctl_extra;
    m.dit->set_input_extension(&m.control_ext);
  } else {
    m.dit->set_input_extension(nullptr);
  }
  if (spec.step_cache != accel::StepCacheKind::kOff) {
    R.cache.emplace(arena, R.h, R.w, static_cast<int>(sh.rows.size()), r.accel.step_cache_threshold);
    m.dit->set_step_cache(&*R.cache);
  } else {
    m.dit->set_step_cache(nullptr);
  }
  for (int bit = 0; bit < kMaxSessionRows; ++bit) {
    if (R.row_bits & (1u << bit)) R.velocities[static_cast<std::size_t>(bit)] = arena.alloc_tensor(DType::kF32, {lat, R.h, R.w});
  }
  if (spec.output != StepOutput::kRows) R.guided = arena.alloc_tensor(DType::kF32, {lat, R.h, R.w});
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  m.sample_memory();

  // The host side: the pinned staging (grown to the largest run), the device-pointer check and the step bookkeeping.
  // Output k's staging slot: none for a row the run lacks (MOSS's fixed role slots: out[1] without an uncond row).
  int slots = 0;
  for (int k = 0; k < R.outputs; ++k) {
    const bool used = spec.output != StepOutput::kRows || (R.row_bits & (1u << k)) != 0;
    R.slot[static_cast<std::size_t>(k)] = used ? slots++ : -1;
  }
  const std::size_t latent = static_cast<std::size_t>(lat * R.h * R.w) * sizeof(float);
  R.layout = runtime::StagingLayout::plan(R.x.nbytes(), slots, latent, R.ctl != nullptr ? R.ctl_extra.nbytes() : 0);
  s.staging.ensure(R.layout.bytes, s.staging_budget());
  R.block = s.staging.block();
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  R.check = std::make_unique<runtime::DevicePointerCheck>(device);
  R.stepped.assign(static_cast<std::size_t>(n), 0);
  s.shape = sh;
  clear.keep = true;
  restore.keep = true;
  s.run = std::move(run);
  s.gate.open_run();
}

RunInfo Session::run_info() const {
  const State& s = *s_;
  if (!s.run) throw std::logic_error("no run is open");
  const Run& R = *s.run;
  RunInfo i;
  i.latent_height = static_cast<int>(R.h);
  i.latent_width = static_cast<int>(R.w);
  i.steps = R.schedule->steps();
  i.outputs = R.outputs;
  i.rows = R.row_bits;
  i.schedule_matches = R.schedule_matches;
  i.off_schedule_calls = R.schedule->off_schedule_calls();
  i.calls = R.schedule->calls();
  i.control_channels = R.ctl != nullptr ? R.ctl->groups() * control::kGroupChannels : 0;
  i.latents = R.latents;
  i.staging_bytes = R.layout.bytes;
  return i;
}

const std::vector<float>& Session::run_sigmas() const {
  if (!s_->run) throw std::logic_error("no run is open");
  return s_->run->schedule->sigmas();
}

StagingViews Session::staging() const {
  if (!s_->run) throw std::logic_error("no run is open");
  const Run& R = *s_->run;
  StagingViews v;
  std::byte* base = R.block.base;
  const std::int64_t lat = moss_dit::kLatentChannels;
  v.x = R.latents == 1 ? Tensor::view(base + R.layout.x, DType::kF32, Device::kCPU, {lat, R.h, R.w})
                       : Tensor::view(base + R.layout.x, DType::kF32, Device::kCPU, {R.latents, lat, R.h, R.w});
  v.outputs = R.outputs;
  for (int k = 0; k < R.outputs; ++k) {
    const int slot = R.slot[static_cast<std::size_t>(k)];
    if (slot < 0) continue;  // a zeroed view
    v.out[static_cast<std::size_t>(k)] =
        Tensor::view(base + R.layout.out[static_cast<std::size_t>(slot)], DType::kF32, Device::kCPU, {lat, R.h, R.w});
  }
  if (R.ctl != nullptr) {
    v.control = Tensor::view(base + R.layout.control, DType::kF32, Device::kCPU, R.ctl_extra.shape.data(), R.ctl_extra.rank);
  }
  return v;
}

StepReport Session::step(const StepArgs& a) {
  State& s = *s_;
  JobGate::Job job = s.gate.begin(JobGate::Kind::kRun);
  Run& R = *s.run;
  Pipeline::Impl& m = s.m;
  State::CallScope scope(m, R.result);
  const auto t0 = Clock::now();
  const cudaStream_t stream = m.streams.compute();
  const RunSpec& spec = R.spec;

  // Position and rows.
  const RunSchedule::Position pos = R.schedule->locate(a.sigma, a.step);
  const int n = R.schedule->steps();
  RowSelection sel;
  std::uint32_t rows = 0;
  if (R.moss) {
    if (a.rows != 0) throw std::invalid_argument("MOSS guidance picks the rows: StepArgs::rows must be 0.");
    sel = select_rows(R.guidance, pos.step, n);
    rows = sel.rows;
  } else {
    rows = a.rows == 0 ? R.row_bits : a.rows;
    if ((rows & ~R.row_bits) != 0) {
      throw std::invalid_argument("The row mask names rows the run does not have (rows 0.." +
                                  std::to_string(spec.rows.size() - 1) + ").");
    }
  }
  const bool guided = spec.output != StepOutput::kRows;

  // Tensors: all device or all host (the staging views), x's grid the run's or a crop of it.
  const bool device = a.x.device == Device::kCUDA;
  const Tensor& x = a.x;
  const std::int64_t lat = moss_dit::kLatentChannels;
  const bool rank3 = x.rank == 3;
  if (x.data == nullptr || x.dtype != DType::kF32 || !x.is_contiguous() || !(x.rank == 4 || (rank3 && R.latents == 1)) ||
      (x.rank == 4 && x.dim(0) != R.latents) || x.dim(-3) != lat || x.dim(-2) < 1 || x.dim(-1) < 1 || x.dim(-2) > R.h ||
      x.dim(-1) > R.w) {
    throw std::invalid_argument("x: expected f32[" + (R.latents == 1 ? std::string() : std::to_string(R.latents) + ", ") +
                                "128, h <= " + std::to_string(R.h) + ", w <= " + std::to_string(R.w) + "], got " +
                                (x.data == nullptr ? std::string("none") : x.describe()));
  }
  const std::int64_t hc = x.dim(-2), wc = x.dim(-1);
  const bool crop = hc != R.h || wc != R.w;
  if (crop && R.moss) throw std::invalid_argument("Crop calls (x smaller than the run's size) need HOST guidance.");
  const StagingViews views = staging();
  const auto same_memory = [&](const Tensor& t, const std::string& what) {
    if ((t.device == Device::kCUDA) != device) throw std::invalid_argument(what + ": tensors must all be device or all host memory.");
  };
  const auto host_slot = [&](const Tensor& t, const Tensor& slot, const std::string& what) {
    if (!device && t.data != slot.data) {
      throw std::invalid_argument(what + ": host memory must be the run's staging view (mi_run_staging).");
    }
  };
  host_slot(x, views.x, "x");
  std::vector<int> out_index;  // the outputs this call writes
  if (guided) {
    out_index.push_back(0);
  } else {
    for (int bit = 0; bit < kMaxSessionRows; ++bit) {
      if (rows & (1u << bit)) out_index.push_back(bit);
    }
  }
  for (int k = 0; k < kMaxSessionRows; ++k) {
    const bool slot = k < R.outputs && R.slot[static_cast<std::size_t>(k)] >= 0;
    if (!slot && a.out[static_cast<std::size_t>(k)].data != nullptr) {
      throw std::invalid_argument("out[" + std::to_string(k) + "]: the run has no such output (pass it zeroed).");
    }
  }
  for (const int k : out_index) {
    const std::string what = "out[" + std::to_string(k) + "]";
    const Tensor& o = a.out[static_cast<std::size_t>(k)];
    expect_tensor(o, DType::kF32, {lat, hc, wc}, what);
    same_memory(o, what);
    host_slot(o, views.out[static_cast<std::size_t>(k)], what);
    if (overlaps(o, x)) throw std::invalid_argument(what + " overlaps x.");
  }
  const int channels = R.ctl != nullptr ? R.ctl->groups() * control::kGroupChannels : 0;
  if (a.control.data != nullptr) {
    if (R.ctl == nullptr) throw std::invalid_argument("control: the session has no control adapter.");
    if (rank3) {
      expect_tensor(a.control, DType::kF32, {channels, hc, wc}, "control");
    } else {
      expect_tensor(a.control, DType::kF32, {R.latents, channels, hc, wc}, "control");
    }
    same_memory(a.control, "control");
    host_slot(a.control, views.control, "control");
  }
  if (device) {
    R.check->check(x.data, x.nbytes(), "x");
    for (const int k : out_index) {
      const Tensor& o = a.out[static_cast<std::size_t>(k)];
      R.check->check(o.data, o.nbytes(), "out[" + std::to_string(k) + "]");
    }
    if (a.control.data != nullptr) R.check->check(a.control.data, a.control.nbytes(), "control");
  }

  // The forward's rows and outputs (a crop packs every device buffer from its start).
  moss_dit::Row frows[kMaxSessionRows];
  Tensor fouts[kMaxSessionRows];
  std::vector<std::int64_t> lengths;
  int nf = 0;
  const auto packed = [&](const Tensor& t) { return Tensor::view(t.data, DType::kF32, Device::kCUDA, {lat, hc, wc}); };
  for (int bit = 0; bit < kMaxSessionRows; ++bit) {
    if (!(rows & (1u << bit))) continue;
    frows[nf] = R.rows[static_cast<std::size_t>(bit)];
    if (R.moss && bit == 1 && sel.uncond_skip) frows[nf].skip = R.guidance.uncond_skip;
    fouts[nf] = device && !guided ? a.out[static_cast<std::size_t>(bit)] : packed(R.velocities[static_cast<std::size_t>(bit)]);
    lengths.push_back(frows[nf].text.dim(0));
    ++nf;
  }
  if (crop) {
    const std::size_t need = m.dit->forward_stack_bytes(hc, wc, lengths);
    if (need > m.arena->available()) {
      throw std::invalid_argument("The crop " + std::to_string(wc * 16) + "x" + std::to_string(hc * 16) + " needs " +
                                  std::to_string(need) + " bytes of forward workspace, more than the " +
                                  std::to_string(m.arena->available()) + " the run planned: begin the run with MI_RUN_CROPS.");
    }
  }
  const Tensor xf = device ? x
                    : rank3 ? Tensor::view(R.x.data, DType::kF32, Device::kCUDA, {lat, hc, wc})
                            : Tensor::view(R.x.data, DType::kF32, Device::kCUDA, {R.latents, lat, hc, wc});
  const float timestep = a.sigma * static_cast<float>(kNumTrainTimesteps);
  const int cached_before = R.cache ? R.cache->reused() : 0;
  struct CropScope {
    moss_dit::MossDit& dit;
    accel::FirstBlockCache* cache;
    bool on;
    ~CropScope() {
      if (!on) return;
      dit.set_step_cache(cache);
      dit.set_graphs_paused(false);
    }
  } crop_scope{*m.dit, R.cache ? &*R.cache : nullptr, crop};
  try {
    if (device) {
      s.handoff->acquire(a.stream, stream);
    } else {
      MI_CUDA_CHECK(cudaMemcpyAsync(xf.data, x.data, x.nbytes(), cudaMemcpyHostToDevice, stream));
    }
    // The control latents: the call's (as given), zeros on a crop, else the run's at this position.
    if (R.ctl != nullptr) {
      const std::int64_t l = R.latents;
      const auto view = [&](void* p, std::int64_t hh, std::int64_t ww) {
        return rank3 ? Tensor::view(p, DType::kF32, Device::kCUDA, {channels, hh, ww})
                     : Tensor::view(p, DType::kF32, Device::kCUDA, {l, channels, hh, ww});
      };
      if (a.control.data != nullptr && device) {
        m.control_ext.extra = a.control;
      } else if (a.control.data != nullptr) {
        MI_CUDA_CHECK(cudaMemcpyAsync(R.ctl_extra.data, a.control.data, a.control.nbytes(), cudaMemcpyHostToDevice, stream));
        m.control_ext.extra = view(R.ctl_extra.data, hc, wc);
        R.ctl_dirty = true;
      } else if (crop) {
        const Tensor zeros = view(R.ctl_extra.data, hc, wc);
        MI_CUDA_CHECK(cudaMemsetAsync(zeros.data, 0, zeros.nbytes(), stream));
        m.control_ext.extra = zeros;
        R.ctl_dirty = true;
      } else {
        const Tensor first = Tensor::view(R.ctl_extra.data, DType::kF32, Device::kCUDA, {channels, R.h, R.w});
        if (R.ctl_dirty) {
          if (l > 1) MI_CUDA_CHECK(cudaMemsetAsync(R.ctl_extra.data, 0, R.ctl_extra.nbytes(), stream));
          R.ctl_in->rewrite(first, stream);
          R.ctl_dirty = false;
        }
        // A control window opened or closed: the DiT's input changed, so the next call runs in full (as mi_generate).
        if (R.ctl_in->apply(first, pos.step, n, stream) > 0 && R.cache) R.cache->invalidate();
        m.control_ext.extra = view(R.ctl_extra.data, R.h, R.w);
      }
    }
    if (a.control_changed && R.cache) R.cache->invalidate();
    if (crop) {
      m.dit->set_step_cache(nullptr);  // crops bypass the step cache, its state untouched
      m.dit->set_graphs_paused(true);  // and CUDA graphs: one-off shapes run eagerly
    }
    m.dit->forward(xf, timestep, std::span<const moss_dit::Row>(frows, static_cast<std::size_t>(nf)),
                   std::span<const Tensor>(fouts, static_cast<std::size_t>(nf)), stream);
    if (guided) {
      const Prediction space = spec.output == StepOutput::kGuidedVelocity ? Prediction::kVelocity : Prediction::kDenoised;
      const Tensor out0 = device ? a.out[0] : R.guided;
      guide_prediction(space, xf, a.sigma, R.velocities[0], sel.cfg() ? R.velocities[1] : Tensor{},
                       sel.slg() ? R.velocities[2] : Tensor{}, sel.cfg(), static_cast<float>(R.guidance.cfg), sel.slg(),
                       static_cast<float>(R.guidance.slg_scale), out0, stream);
      if (!device) MI_CUDA_CHECK(cudaMemcpyAsync(a.out[0].data, R.guided.data, a.out[0].nbytes(), cudaMemcpyDeviceToHost, stream));
    } else if (!device) {
      for (int f = 0; f < nf; ++f) {
        const Tensor& o = a.out[static_cast<std::size_t>(out_index[static_cast<std::size_t>(f)])];
        MI_CUDA_CHECK(cudaMemcpyAsync(o.data, fouts[f].data, o.nbytes(), cudaMemcpyDeviceToHost, stream));
      }
    }
    if (device) {
      s.handoff->release(stream, a.stream);
    } else {
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  } catch (const moss_dit::Cancelled&) {
    sync_quietly(stream);
    if (R.cache) R.cache->invalidate();
    throw SessionCancelled("the step was cancelled");
  } catch (...) {
    sync_quietly(stream);
    throw;
  }
  R.schedule->commit(pos);
  R.stepped[static_cast<std::size_t>(pos.step)] = 1;
  StepReport rep;
  rep.step = pos.step;
  rep.rows = rows;
  rep.cached = !crop && R.cache && R.cache->reused() > cached_before;
  rep.on_schedule = pos.on_schedule;
  rep.seconds = since(t0);
  if (rep.cached) {
    std::vector<int>& cs = R.result.accel.cached_steps;
    if (cs.empty() || cs.back() != pos.step + 1) cs.push_back(pos.step + 1);
  }
  R.result.timings.step_s.push_back(rep.seconds);
  return rep;
}

RunResult Session::State::finish_run() {
  std::unique_ptr<Run> owned = std::move(run);
  Run& R = *owned;
  Result& r = R.result;
  CallScope scope(m, r);
  sync_quietly(m.streams.compute());
  clear_dit();
  const int n = R.schedule->steps();
  if (R.cache) {
    r.accel.dit_calls = R.cache->calls();
    r.accel.cached_calls = R.cache->reused();
    r.accel.step_cache_diffs = R.cache->diffs();
  }
  for (int i = 0; i < n; ++i) {
    r.accel.cfg_truncated_steps += R.stepped[static_cast<std::size_t>(i)] && R.guidance.do_cfg() && !R.guidance.cfg_active(i, n);
  }
  r.height = R.spec.height;
  r.width = R.spec.width;
  r.steps_scheduled = n;
  r.steps_run = static_cast<int>(std::count(R.stepped.begin(), R.stepped.end(), 1));
  r.timings.total_s = since(R.started);
  finish_report(r);
  if (m.learner) m.learner->end_generation();
  RunResult out;
  out.guidance = R.spec.guidance;
  out.schedule_matches = R.schedule_matches;
  out.off_schedule_calls = R.schedule->off_schedule_calls();
  out.calls = R.schedule->calls();
  for (std::size_t k = 0; k < R.spec.rows.size(); ++k) {
    if (!R.spec.rows[k].conditioning->engine_encoded) out.foreign_rows.push_back(static_cast<int>(k));
  }
  out.note = R.schedule_note;
  if (out.off_schedule_calls > 0) {
    out.note += (out.note.empty() ? "" : " ") + std::to_string(out.off_schedule_calls) + " of " + std::to_string(out.calls) +
                " calls had a sigma off the run's schedule.";
  }
  if (!out.foreign_rows.empty()) {
    std::string list;
    for (const int k : out.foreign_rows) list += (list.empty() ? "" : ", ") + std::to_string(k);
    out.note += (out.note.empty() ? "" : " ") + std::string("Foreign conditioning (not encoded by this engine) on rows ") + list + ".";
  }
  out.result = std::move(r);
  owned.reset();  // the run's stack, step cache and control latents
  return out;
}

std::optional<RunResult> Session::end_run() {
  State& s = *s_;
  std::optional<RunResult> out;
  s.gate.end_run([&] { out = s.finish_run(); });
  return out;
}

void Session::vae_encode(const Image& image, const Tensor& latents) {
  State& s = *s_;
  JobGate::Job job = s.gate.begin(JobGate::Kind::kSession);
  const int H = image.height, W = image.width;
  if (image.channels != 3 || H <= 0 || W <= 0 || H % 16 != 0 || W % 16 != 0 || H > kMaxSide || W > kMaxSide ||
      std::int64_t{H} * W > kMaxPixels || image.pixels.size() != static_cast<std::size_t>(H) * W * 3) {
    throw std::invalid_argument("The image must be RGB8 with sides multiples of 16, at most " + std::to_string(kMaxSide) +
                                " on a side and 4.2 megapixels.");
  }
  expect_tensor(latents, DType::kF32, {moss_dit::kLatentChannels, H / 16, W / 16}, "latents");
  if (latents.device == Device::kCUDA) {
    int device = 0;
    MI_CUDA_CHECK(cudaGetDevice(&device));
    runtime::DevicePointerCheck(device).check(latents.data, latents.nbytes(), "latents");
  }
  Pipeline::Impl& m = s.m;
  Result r;
  s.start_report(r);
  State::CallScope scope(m, r);
  s.enter(Phase::kDecode);
  const Pipeline::Impl::CallModels models = s.models_for(s.request, r);
  if (!s.shape) s.shape = s.request_shape(*models.b);
  const Pipeline::Impl::CallDit d = m.setup_dit_call(*models.src, r);
  const EncoderSource& enc = m.encoder_source(*models.b, r.timings.index_load_s);
  std::size_t denoise = 0;
  s.apply_plan(models, s.plan(models, d, *s.shape, H, W, &enc, {}, denoise, r), r);
  PassReport pass;
  const std::vector<float> z = m.encode_image(enc, image, pass, r.timings);
  MI_CUDA_CHECK(cudaMemcpy(latents.data, z.data(), latents.nbytes(), cudaMemcpyDefault));
  s.finish_report(r);
}

void Session::vae_decode(const Tensor& latents, const Tensor& rgb) {
  State& s = *s_;
  JobGate::Job job = s.gate.begin(JobGate::Kind::kSession);
  if (latents.rank != 3) throw std::invalid_argument("latents: expected f32[128, h, w], got " + latents.describe());
  const std::int64_t h = latents.dim(1), w = latents.dim(2);
  expect_tensor(latents, DType::kF32, {moss_dit::kLatentChannels, h, w}, "latents");
  const int H = static_cast<int>(h * 16), W = static_cast<int>(w * 16);
  if (H > kMaxSide || W > kMaxSide || std::int64_t{H} * W > kMaxPixels) {
    throw std::invalid_argument("latents: the image would exceed " + std::to_string(kMaxSide) + " on a side or 4.2 megapixels.");
  }
  expect_tensor(rgb, DType::kU8, {H, W, 3}, "rgb");
  if (latents.device == Device::kCUDA || rgb.device == Device::kCUDA) {
    int device = 0;
    MI_CUDA_CHECK(cudaGetDevice(&device));
    const runtime::DevicePointerCheck check(device);
    if (latents.device == Device::kCUDA) check.check(latents.data, latents.nbytes(), "latents");
    if (rgb.device == Device::kCUDA) check.check(rgb.data, rgb.nbytes(), "rgb");
  }
  Pipeline::Impl& m = s.m;
  Result r;
  s.start_report(r);
  State::CallScope scope(m, r);
  s.enter(Phase::kDecode);
  const Pipeline::Impl::CallModels models = s.models_for(s.request, r);
  if (!s.shape) s.shape = s.request_shape(*models.b);
  const Pipeline::Impl::CallDit d = m.setup_dit_call(*models.src, r);
  std::size_t denoise = 0;
  s.apply_plan(models, s.plan(models, d, *s.shape, H, W, nullptr, {}, denoise, r), r);
  std::vector<float> lat(static_cast<std::size_t>(latents.numel()));
  MI_CUDA_CHECK(cudaMemcpy(lat.data(), latents.data, latents.nbytes(), cudaMemcpyDefault));
  m.ensure_vae(*models.vs, r.plan.vae.stack_bytes, r.plan.dit.park_for_decode, r.timings);
  std::vector<std::uint8_t> out;
  try {
    m.decode_image(lat, H, W, out, s.gate.cancel_flag(), r.timings);
  } catch (const flux2_vae::Cancelled&) {
    if (m.vae) m.vae->set_cancel(nullptr);
    throw SessionCancelled("the decode was cancelled");
  }
  if (m.vae) m.vae->set_cancel(nullptr);
  MI_CUDA_CHECK(cudaMemcpy(rgb.data, out.data(), rgb.nbytes(), cudaMemcpyDefault));
  s.finish_report(r);
}

}  // namespace mi::pipeline
