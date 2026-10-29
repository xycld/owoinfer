#include "pipeline/nvfp4_calibration.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>

#include "core/cuda_check.h"
#include "kernels/basic/gram.h"
#include "pipeline/quality_report.h"
#include "util/xxh64.h"

namespace mi::pipeline {
namespace {

struct JobSpec {
  const char* id;
  long long seed;
  double cfg, shift;
  const char* uncond_skip;  // comma-separated block indices
  const char* prompt;
  const char* negative;
};

constexpr JobSpec kJobs[] = {
#include "pipeline/nvfp4_calibration_set.inc"
};

constexpr int kCalibSize = 1024;
constexpr int kCalibSteps = 50;

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

std::string jstr(std::string_view s) { return "\"" + std::string(s) + "\""; }  // ASCII identifiers only

quant::GemmUnit unit_of(moss_dit::BlockGemm g) {
  switch (g) {
    case moss_dit::BlockGemm::kQkv: return quant::GemmUnit::kQkv;
    case moss_dit::BlockGemm::kO: return quant::GemmUnit::kO;
    case moss_dit::BlockGemm::kUpGate: return quant::GemmUnit::kUpGate;
    case moss_dit::BlockGemm::kDown: return quant::GemmUnit::kDown;
  }
  return quant::GemmUnit::kQkv;
}

std::string numerics_object(const EngineNumerics& n) {
  std::string s = "{";
  for (std::size_t i = 0; i < n.size(); ++i) s += (i == 0 ? "" : ", ") + jstr(n[i].first) + ": " + std::to_string(n[i].second);
  return s + "}";
}

}  // namespace

std::vector<CalibrationJob> nvfp4_calibration_jobs(const std::filesystem::path& model) {
  std::vector<CalibrationJob> jobs;
  for (const JobSpec& j : kJobs) {
    CalibrationJob c;
    c.id = j.id;
    Request& r = c.request;
    r.model = model;
    r.prompt = j.prompt;
    r.negative_prompt = j.negative;
    r.height = r.width = kCalibSize;
    r.steps = kCalibSteps;
    r.cfg = j.cfg;
    r.shift = j.shift;
    r.seed = j.seed;
    r.uncond_skip_layers = parse_layer_list(j.uncond_skip);
    r.slg_layers.clear();
    r.max_sequence_length = 4096;
    r.sampler = SamplerKind::kEuler;
    r.scheduler = SchedulerKind::kMoss;
    r.decode = false;
    r.save_metadata = false;
    jobs.push_back(std::move(c));
  }
  return jobs;
}

const std::vector<int>& nvfp4_calibration_steps() {
  static const std::vector<int> steps = {6, 18, 30, 43};
  return steps;
}

std::uint64_t nvfp4_calibration_set_xxh64() {
  Xxh64 h;
  auto add = [&](std::string_view s) {
    h.update(s.data(), s.size());
    h.update("\0", 1);
  };
  add(kNvfp4CalibrationSetId);
  // Numbers by std::to_chars (the shortest round-trip form), which no locale changes.
  auto num = [](auto v) {
    char b[64];
    return std::string(b, std::to_chars(b, b + sizeof b, v).ptr);
  };
  for (const JobSpec& j : kJobs) {
    add(j.id);
    add(num(kCalibSize) + "x" + num(kCalibSize) + " steps=" + num(kCalibSteps) + " seed=" + num(j.seed) + " cfg=" + num(j.cfg) +
        " shift=" + num(j.shift) + " skip=" + j.uncond_skip);
    add(j.prompt);
    add(j.negative);
  }
  for (int s : nvfp4_calibration_steps()) add(std::to_string(s));
  return h.digest();
}

quant::HessianSource nvfp4_hessian_source(CalibrationOptions options) {
  return [options = std::move(options)](const std::vector<quant::HessianUnit>& units, const quant::HessianSink& sink) {
    auto log = [&](const std::string& s) {
      if (options.log) options.log(s);
    };
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      (void)cudaGetLastError();
      throw std::runtime_error("the gptq recipes capture calibration Hessians on a CUDA GPU, and none is usable (use --recipe mse or rtn)");
    }
    // Units in (block, unit) order: the order the passes and the sink follow.
    std::vector<std::size_t> order(units.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return std::pair(units[a].block, units[a].unit) < std::pair(units[b].block, units[b].unit);
    });
    std::size_t budget = options.hessian_bytes;
    if (budget == 0) {
      std::size_t free_b = 0, total_b = 0;
      MI_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      budget = free_b > kCalibrationHeadroom ? free_b - kCalibrationHeadroom : 0;
    }
    // Passes: whole blocks, as many as fit the budget (at least one block).
    std::vector<std::vector<std::size_t>> passes;
    {
      std::size_t used = 0;
      for (std::size_t i = 0; i < order.size();) {
        std::size_t j = i, block_bytes = 0;
        for (; j < order.size() && units[order[j]].block == units[order[i]].block; ++j) {
          block_bytes += static_cast<std::size_t>(units[order[j]].k * units[order[j]].k) * sizeof(float);
        }
        if (passes.empty() || used + block_bytes > budget) {
          passes.emplace_back();
          used = 0;
        }
        for (std::size_t t = i; t < j; ++t) passes.back().push_back(order[t]);
        used += block_bytes;
        i = j;
      }
    }

    // The capture's strings come first, so their size is checked before any work: the Hessians' digest is the
    // converter's (over exactly the units it uses), and the text encoder must be the bf16 one capture_class names.
    const std::vector<int>& steps = nvfp4_calibration_steps();
    const std::string gpu_arch = current_gpu_arch();
    quant::HessianCapture cap;
    {
      std::string steps_json, ids;
      for (std::size_t i = 0; i < steps.size(); ++i) steps_json += (i ? ", " : "") + std::to_string(steps[i]);
      for (std::size_t i = 0; i < std::size(kJobs); ++i) ids += (i ? ", " : "") + jstr(kJobs[i].id);
      cap.calibration = "{\"set\": " + jstr(kNvfp4CalibrationSetId) + ", \"set_xxh64\": " + jstr(hex64(nvfp4_calibration_set_xxh64())) +
                        ", \"prompts\": [" + ids + "], \"size\": [" + std::to_string(kCalibSize) + ", " + std::to_string(kCalibSize) +
                        "], \"steps\": " + std::to_string(kCalibSteps) + ", \"capture_steps\": [" + steps_json + "], \"rows\": \"cond and uncond\"}";
      cap.capture_class = "{\"gpu_arch\": " + jstr(gpu_arch) + ", \"te\": \"bf16\", \"numerics\": " + numerics_object(engine_numerics(RunConfig{})) +
                          ", \"gram\": " + std::to_string(kernels::kGramNumericsVersion) + ", \"dit_blocks\": \"bf16 op_by_op\"}";
      if (cap.calibration.size() > quant::kMaxCaptureStringBytes || cap.capture_class.size() > quant::kMaxCaptureStringBytes) {
        throw std::logic_error("calibration: capture strings longer than quant::kMaxCaptureStringBytes");
      }
    }

    PipelineOptions po;
    po.block_path = moss_dit::BlockPath::kOpByOp;  // the hook's path; bit-identical to the fused blocks
    po.tier = planner::Tier::kReference;
    po.attention = planner::AttentionMode::kExact;
    po.dit_file = options.dit_file;
    Pipeline pipe(po);
    const std::vector<CalibrationJob> jobs = nvfp4_calibration_jobs(options.model);
    const int last_step = steps.back();
    log("calibration: " + std::to_string(jobs.size()) + " prompts at " + std::to_string(kCalibSize) + "x" + std::to_string(kCalibSize) +
        ", DiT inputs of steps 6/18/30/43, " + std::to_string(units.size()) + " Hessians in " + std::to_string(passes.size()) + " pass(es)");

    std::vector<float> host;
    for (std::size_t p = 0; p < passes.size(); ++p) {
      const auto tp = std::chrono::steady_clock::now();
      // The pass's Hessians, zeroed (the lower tiles stay zero).
      std::map<std::pair<int, quant::GemmUnit>, DeviceBuffer> h;
      for (std::size_t u : passes[p]) {
        const std::size_t bytes = static_cast<std::size_t>(units[u].k * units[u].k) * sizeof(float);
        DeviceBuffer b(bytes);
        MI_CUDA_CHECK(cudaMemset(b.get(), 0, bytes));
        h.emplace(std::pair(units[u].block, units[u].unit), std::move(b));
      }
      MI_CUDA_CHECK(cudaDeviceSynchronize());  // the memsets (legacy stream) before any Gram kernel (the pipeline's stream)
      int step = 0;  // the step the next DiT forward belongs to
      GenerateControl control;
      control.on_step = [&](const StepInfo& info) {
        step = info.step;
        return info.step <= last_step;  // stop once the last capture step has run
      };
      control.dit_linear_inputs = [&](int block, moss_dit::BlockGemm g, const Tensor& x, cudaStream_t stream) {
        if (!std::binary_search(steps.begin(), steps.end(), step)) return;
        const auto it = h.find(std::pair(block, unit_of(g)));
        if (it != h.end()) kernels::gram_accumulate_upper(x, static_cast<float*>(it->second.get()), stream);
      };
      for (const CalibrationJob& job : jobs) {
        step = 0;
        const Result r = pipe.generate(job.request, control);
        if (r.steps_run != last_step + 1) {
          throw std::runtime_error("calibration: " + job.id + " ran " + std::to_string(r.steps_run) + " steps, expected " +
                                   std::to_string(last_step + 1));
        }
        if (r.dit_linears != "bf16") throw std::runtime_error("calibration needs a bf16 DiT, got " + r.dit_linears);
        if (r.te_precision != "bf16") {
          throw std::runtime_error("calibration needs the bundle's bf16 text encoder, got " + r.te_precision + " (" + r.te_source + ")");
        }
        if (r.gpu_arch != gpu_arch) throw std::runtime_error("calibration: the pipeline ran on " + r.gpu_arch + ", expected " + gpu_arch);
      }
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      log("  pass " + std::to_string(p + 1) + "/" + std::to_string(passes.size()) + ": blocks " + std::to_string(units[passes[p].front()].block) + "-" +
          std::to_string(units[passes[p].back()].block) + " captured in " +
          std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count()) + " s");
      for (std::size_t u : passes[p]) {
        DeviceBuffer& b = h.at(std::pair(units[u].block, units[u].unit));
        host.resize(static_cast<std::size_t>(units[u].k * units[u].k));
        b.download(host.data(), host.size() * sizeof(float));
        b = DeviceBuffer();
        sink(u, host.data());
      }
    }

    if (gpu_arch != "sm_120") {
      log("note: captured on " + gpu_arch + "; the published NVFP4 conversions are captured on sm_120 (the only class that runs NVFP4)");
    }
    return cap;
  };
}

}  // namespace mi::pipeline
