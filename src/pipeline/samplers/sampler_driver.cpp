#include "pipeline/samplers/sampler_driver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/cuda_check.h"
#include "pipeline/samplers/brownian_tree.h"
#include "pipeline/samplers/sampler_kernels.h"
#include "pipeline/samplers/torch_cuda_randn.h"
#include "weights/file_reader.h"
#include "weights/safetensors.h"

namespace mi::pipeline {
namespace sk = sampler_kernels;
namespace {

constexpr SamplerKind kAllSamplers[] = {SamplerKind::kEuler,        SamplerKind::kAb2,    SamplerKind::kEulerAncestral,
                                        SamplerKind::kHeun,         SamplerKind::kDpmpp2m, SamplerKind::kDpmpp2mSde,
                                        SamplerKind::kErSde,        SamplerKind::kResMultistep, SamplerKind::kDataSpace};

void check_tensor(const Tensor& t, std::int64_t numel, const char* who) {
  if (t.device != Device::kCUDA || t.dtype != DType::kF32 || !t.is_contiguous() || t.numel() != numel) {
    throw std::invalid_argument(std::string(who) + ": expected contiguous CUDA fp32 with " + std::to_string(numel) +
                                " elements, got " + t.describe());
  }
}

// Samplers whose step-size ratios divide by zero on a zero-length step.
bool rejects_repeated_sigmas(SamplerKind k) {
  return k == SamplerKind::kAb2 || k == SamplerKind::kDpmpp2m || k == SamplerKind::kDpmpp2mSde ||
         k == SamplerKind::kErSde || k == SamplerKind::kResMultistep;
}

bool uses_coefficients(SamplerKind k, bool velocity_euler) {
  return !(k == SamplerKind::kAb2 || k == SamplerKind::kDataSpace || (k == SamplerKind::kEuler && velocity_euler));
}

sk::CoefKind coef_kind(SamplerKind k) {
  switch (k) {
    case SamplerKind::kEulerAncestral: return sk::CoefKind::kAncestral;
    case SamplerKind::kHeun: return sk::CoefKind::kHeun;
    case SamplerKind::kDpmpp2m: return sk::CoefKind::kDpm2m;
    case SamplerKind::kDpmpp2mSde: return sk::CoefKind::kSde;
    case SamplerKind::kErSde: return sk::CoefKind::kEr;
    case SamplerKind::kResMultistep: return sk::CoefKind::kRes;
    default: return sk::CoefKind::kEuler;
  }
}

// Latent-sized scratch buffers (beyond the model prediction) each algorithm keeps.
int extra_buffers(SamplerKind k) {
  switch (k) {
    case SamplerKind::kAb2:
    case SamplerKind::kDpmpp2m:
    case SamplerKind::kResMultistep: return 1;            // previous prediction
    case SamplerKind::kEulerAncestral: return 1;          // noise
    case SamplerKind::kHeun: return 2;                    // slope, predictor
    case SamplerKind::kDpmpp2mSde: return 2;              // previous prediction, noise
    case SamplerKind::kErSde: return 4;                   // previous prediction, noise, two divided differences
    case SamplerKind::kDataSpace: return 2;               // initial noise, model input
    case SamplerKind::kEuler: return 0;
  }
  return 0;
}

template <typename T>
std::vector<float> read_floats(const weights::FileReader& in, const weights::TensorInfo& t) {
  std::vector<T> raw(static_cast<std::size_t>(t.numel()));
  try {
    in.read_at(t.offset, raw.data(), raw.size() * sizeof(T));
  } catch (const weights::IoError&) {
    throw DataspaceError("load_dataspace: cannot read tensor " + t.name);
  }
  return std::vector<float>(raw.begin(), raw.end());
}

}  // namespace

const char* sampler_name(SamplerKind kind) {
  switch (kind) {
    case SamplerKind::kEuler: return "euler";
    case SamplerKind::kAb2: return "ab2";
    case SamplerKind::kEulerAncestral: return "euler_ancestral";
    case SamplerKind::kHeun: return "heun";
    case SamplerKind::kDpmpp2m: return "dpmpp_2m";
    case SamplerKind::kDpmpp2mSde: return "dpmpp_2m_sde";
    case SamplerKind::kErSde: return "er_sde";
    case SamplerKind::kResMultistep: return "res_multistep";
    case SamplerKind::kDataSpace: return "dataspace";
  }
  return "?";
}

SamplerKind parse_sampler(std::string_view name) {
  for (const SamplerKind k : kAllSamplers) {
    if (name == sampler_name(k)) return k;
  }
  throw std::invalid_argument("unknown sampler '" + std::string(name) +
                              "' (expected euler, ab2, euler_ancestral, heun, dpmpp_2m, dpmpp_2m_sde, er_sde, "
                              "res_multistep or dataspace)");
}

bool sampler_is_stochastic(SamplerKind kind) {
  return kind == SamplerKind::kEulerAncestral || kind == SamplerKind::kDpmpp2mSde || kind == SamplerKind::kErSde;
}

bool sampler_needs_options(SamplerKind kind) { return sampler_is_stochastic(kind) || kind == SamplerKind::kDataSpace; }

int model_calls_per_step(SamplerKind kind) { return kind == SamplerKind::kHeun ? 2 : 1; }

const char* noise_source_name(NoiseSource source) {
  switch (source) {
    case NoiseSource::kPhilox: return "philox";
    case NoiseSource::kPhiloxDevice: return "philox_device";
    case NoiseSource::kBrownianTree: return "brownian_tree";
  }
  return "?";
}

NoiseSource parse_noise_source(std::string_view name) {
  for (const NoiseSource s : {NoiseSource::kPhilox, NoiseSource::kPhiloxDevice, NoiseSource::kBrownianTree}) {
    if (name == noise_source_name(s)) return s;
  }
  throw std::invalid_argument("unknown noise source '" + std::string(name) +
                              "' (expected philox, philox_device or brownian_tree)");
}

DataSpaceParams load_dataspace(const std::string& path) {
  std::optional<weights::SafetensorsFile> opened;
  try {
    opened.emplace(util::path_from_utf8(path));
  } catch (const weights::FormatError& e) {
    throw DataspaceError(e.what());
  }
  const weights::SafetensorsFile& file = *opened;
  const weights::TensorInfo* q = file.find("latent_mix.Q");
  const weights::TensorInfo* codes = file.find("step_codes");
  if (q == nullptr || codes == nullptr) throw DataspaceError(path + ": expected tensors latent_mix.Q and step_codes");
  if (q->dtype != weights::DType::kF32 || q->shape.size() != 2 || q->shape[0] != q->shape[1] || q->shape[0] < 1 ||
      q->shape[0] > 4096) {
    throw DataspaceError(path + ": latent_mix.Q must be F32 [C, C]");
  }
  if (codes->shape.size() != 1 || codes->shape[0] < 1 || codes->shape[0] > 1000) {
    throw DataspaceError(path + ": step_codes must be a vector of 1..1000 values");
  }
  const weights::FileReader in(file.path());  // regular files only, as SafetensorsFile opened it
  DataSpaceParams p;
  p.channels = static_cast<int>(q->shape[0]);
  p.q = read_floats<float>(in, *q);
  switch (codes->dtype) {
    case weights::DType::kF32: p.step_codes = read_floats<float>(in, *codes); break;
    case weights::DType::kF64: p.step_codes = read_floats<double>(in, *codes); break;
    case weights::DType::kI64: p.step_codes = read_floats<std::int64_t>(in, *codes); break;
    default: throw DataspaceError(path + ": step_codes must be F32, F64 or I64");
  }
  for (const float c : p.step_codes) {
    if (!std::isfinite(c) || c < 0 || c > 1000) throw DataspaceError(path + ": step codes must lie in [0, 1000]");
  }
  for (const float v : p.q) {
    if (!std::isfinite(v)) throw DataspaceError(path + ": latent_mix.Q is not finite");
  }
  return p;
}

std::vector<float> dataspace_sigmas(const DataSpaceParams& params, int steps) {
  if (steps < 1 || static_cast<std::size_t>(steps) > params.step_codes.size()) {
    throw std::invalid_argument("dataspace: steps must be between 1 and the number of step codes (" +
                                std::to_string(params.step_codes.size()) + ")");
  }
  std::vector<float> s;
  for (int i = 0; i < steps; ++i) s.push_back(params.step_codes[static_cast<std::size_t>(i)] / 1000.0f);
  s.push_back(0.0f);
  return s;
}

struct SamplerDriver::Impl {
  SamplerKind kind{};
  SamplerOptions options;
  DataSpaceParams dataspace;  // own copy
  std::vector<float> sigmas;
  std::vector<std::int64_t> shape;
  std::int64_t numel = 0;
  float sigma_min = 0, sigma_max = 0;  // of the schedule before the SDE offset (Brownian tree interval)
  std::vector<float> coef;
  bool official_euler = false;
  bool ran = false;

  Tensor view(void* p) const { return Tensor::view(p, DType::kF32, Device::kCUDA, shape.data(), static_cast<int>(shape.size())); }
};

SamplerDriver::SamplerDriver(SamplerKind kind, std::vector<float> sigmas, std::vector<std::int64_t> latent_shape,
                             const SamplerOptions& options, bool official_euler)
    : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.kind = kind;
  m.options = options;
  m.official_euler = official_euler && kind == SamplerKind::kEuler;
  m.options.dataspace = nullptr;
  if (sigmas.size() < 2) throw std::invalid_argument("SamplerDriver: the schedule needs at least 2 sigmas");
  for (std::size_t i = 0; i < sigmas.size(); ++i) {
    if (!std::isfinite(sigmas[i]) || sigmas[i] < 0 || (i > 0 && sigmas[i] > sigmas[i - 1])) {
      throw std::invalid_argument("SamplerDriver: sigmas must be finite, non-negative and non-increasing");
    }
  }
  if (rejects_repeated_sigmas(kind)) {
    for (std::size_t i = 1; i < sigmas.size(); ++i) {
      if (sigmas[i] > 0 && sigmas[i] == sigmas[i - 1]) {
        throw std::invalid_argument(std::string("SamplerDriver: sampler ") + sampler_name(kind) +
                                    " needs strictly decreasing sigmas, but sigma " + std::to_string(i) +
                                    " repeats the previous one (a table-based schedule with more than 1000 steps?)");
      }
    }
  }
  if (latent_shape.empty() || latent_shape.size() > static_cast<std::size_t>(Tensor::kMaxRank)) {
    throw std::invalid_argument("SamplerDriver: bad latent shape");
  }
  m.numel = 1;
  for (const std::int64_t d : latent_shape) {
    if (d < 1) throw std::invalid_argument("SamplerDriver: bad latent shape");
    m.numel *= d;
  }
  m.shape = std::move(latent_shape);
  if (!std::isfinite(options.eta) || options.eta < 0 || !std::isfinite(options.s_noise) || options.s_noise < 0) {
    throw std::invalid_argument("SamplerDriver: eta and s_noise must be finite and non-negative");
  }
  if (kind == SamplerKind::kDataSpace) {
    if (options.dataspace == nullptr) throw std::invalid_argument("SamplerDriver: dataspace sampler needs DataSpaceParams");
    m.dataspace = *options.dataspace;
    const auto c = static_cast<std::size_t>(m.dataspace.channels);
    if (c == 0 || m.dataspace.q.size() != c * c || m.shape[0] != m.dataspace.channels) {
      throw std::invalid_argument("SamplerDriver: dataspace Q must be [C, C] with C the latent's first dimension");
    }
    if (sigmas.size() - 1 > m.dataspace.step_codes.size()) {
      throw std::invalid_argument("SamplerDriver: more dataspace steps than step codes");
    }
  }
  m.sigma_max = sigmas.front();
  m.sigma_min = sigmas.front();
  for (const float s : sigmas) {
    if (s > 0) m.sigma_min = std::min(m.sigma_min, s);
  }
  // offset_first_sigma_for_snr: the flow log-SNR is infinite at sigma 1.
  if ((kind == SamplerKind::kDpmpp2mSde || kind == SamplerKind::kErSde) && sigmas[0] >= 1.0f) {
    sigmas[0] = static_cast<float>(flow_percent_to_sigma(options.flow, 1e-4));
  }
  m.sigmas = std::move(sigmas);
}

SamplerDriver::~SamplerDriver() = default;
SamplerDriver::SamplerDriver(SamplerDriver&&) noexcept = default;
SamplerDriver& SamplerDriver::operator=(SamplerDriver&&) noexcept = default;

SamplerKind SamplerDriver::kind() const { return impl_->kind; }

Prediction SamplerDriver::prediction() const {
  const SamplerKind k = impl_->kind;
  const bool velocity = k == SamplerKind::kAb2 || k == SamplerKind::kDataSpace || impl_->official_euler;
  return velocity ? Prediction::kVelocity : Prediction::kDenoised;
}

int SamplerDriver::num_steps() const { return static_cast<int>(impl_->sigmas.size()) - 1; }

int SamplerDriver::num_model_calls() const {
  int calls = num_steps();
  if (impl_->kind == SamplerKind::kHeun) {
    for (std::size_t i = 1; i < impl_->sigmas.size(); ++i) calls += impl_->sigmas[i] != 0.0f;
  }
  return calls;
}

const std::vector<float>& SamplerDriver::sigmas() const { return impl_->sigmas; }
const std::vector<float>& SamplerDriver::coefficients() const { return impl_->coef; }

int SamplerDriver::coefficients_per_step(SamplerKind kind) {
  switch (kind) {
    case SamplerKind::kEuler: return sk::kEulCount;
    case SamplerKind::kEulerAncestral: return sk::kAncCount;
    case SamplerKind::kHeun: return sk::kHeuCount;
    case SamplerKind::kDpmpp2m: return sk::kDpmCount;
    case SamplerKind::kDpmpp2mSde: return sk::kSdeCount;
    case SamplerKind::kErSde: return sk::kErCount;
    case SamplerKind::kResMultistep: return sk::kResCount;
    case SamplerKind::kAb2:
    case SamplerKind::kDataSpace: return 0;
  }
  return 0;
}

std::size_t SamplerDriver::workspace_bytes(SamplerKind kind, std::int64_t numel, int steps, int channels) {
  const std::size_t latent = runtime::align_up(static_cast<std::size_t>(numel) * sizeof(float));
  std::size_t bytes = latent * static_cast<std::size_t>(1 + extra_buffers(kind));
  const auto s = static_cast<std::size_t>(std::max(steps, 0));
  bytes += runtime::align_up((s + 1) * sizeof(float));
  bytes += runtime::align_up(s * static_cast<std::size_t>(std::max(coefficients_per_step(kind), 1)) * sizeof(float));
  if (kind == SamplerKind::kDataSpace) {
    bytes += runtime::align_up(static_cast<std::size_t>(channels) * static_cast<std::size_t>(channels) * sizeof(float));
  }
  return bytes;
}

int SamplerDriver::run(const Tensor& x, const ModelFn& model, runtime::DeviceArena& arena, cudaStream_t stream,
                       const StepFn& after_step) {
  Impl& m = *impl_;
  check_tensor(x, m.numel, "SamplerDriver::run");
  if (m.ran) throw std::logic_error("SamplerDriver::run: a driver runs once");
  m.ran = true;
  const SamplerKind kind = m.kind;
  const std::int64_t n = m.numel;
  const int steps = num_steps();
  const bool velocity_euler = m.official_euler;

  runtime::StackScope scope(arena);
  const std::size_t latent_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const auto buffer = [&] { return static_cast<float*>(arena.alloc_stack(latent_bytes)); };
  float* pred = buffer();
  float* old = nullptr;
  float* noise = nullptr;
  float* aux1 = nullptr;
  float* aux2 = nullptr;
  switch (kind) {
    case SamplerKind::kAb2:
    case SamplerKind::kDpmpp2m:
    case SamplerKind::kResMultistep: old = buffer(); break;
    case SamplerKind::kEulerAncestral: noise = buffer(); break;
    case SamplerKind::kHeun:
    case SamplerKind::kDataSpace:
      aux1 = buffer();
      aux2 = buffer();
      break;
    case SamplerKind::kDpmpp2mSde:
      old = buffer();
      noise = buffer();
      break;
    case SamplerKind::kErSde:
      old = buffer();
      noise = buffer();
      aux1 = buffer();
      aux2 = buffer();
      break;
    case SamplerKind::kEuler: break;
  }

  // Coefficients, evaluated on the device as the reference's 0-dim CUDA tensor ops.
  const int per_step = coefficients_per_step(kind);
  m.coef.clear();
  if (uses_coefficients(kind, velocity_euler)) {
    auto* d_sigmas = static_cast<float*>(arena.alloc_stack(m.sigmas.size() * sizeof(float)));
    auto* d_coef = static_cast<float*>(arena.alloc_stack(static_cast<std::size_t>(steps * per_step) * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpyAsync(d_sigmas, m.sigmas.data(), m.sigmas.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    sk::compute_coefficients(coef_kind(kind), d_sigmas, steps, static_cast<float>(m.options.eta),
                             static_cast<float>(m.options.eta + 1.0), d_coef, stream);
    m.coef.resize(static_cast<std::size_t>(steps * per_step));
    MI_CUDA_CHECK(cudaMemcpyAsync(m.coef.data(), d_coef, m.coef.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Noise source.
  std::unique_ptr<TorchCudaGenerator> philox;
  std::unique_ptr<BrownianTreeNoise> brownian;
  std::vector<float> host_noise;
  if (sampler_is_stochastic(kind) && m.options.noise != NoiseSource::kBrownianTree) {
    philox = std::make_unique<TorchCudaGenerator>(
        m.options.seed, m.options.noise == NoiseSource::kPhiloxDevice ? CudaRandnGrid::kDevice : CudaRandnGrid::kUncapped);
  }
  const auto draw_noise = [&](int i) {
    if (!philox && !brownian) {
      // Built at the first draw: a schedule that never adds noise (one step into 0) needs no tree.
      brownian = std::make_unique<BrownianTreeNoise>(n, m.sigma_min, m.sigma_max, m.options.seed);
    }
    if (philox) {
      philox->draw(m.view(noise), stream);
    } else {
      host_noise = brownian->noise(m.sigmas[static_cast<std::size_t>(i)], m.sigmas[static_cast<std::size_t>(i) + 1]);
      MI_CUDA_CHECK(cudaMemcpyAsync(noise, host_noise.data(), latent_bytes, cudaMemcpyHostToDevice, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));  // host_noise is reused next step
    }
  };

  float* q = nullptr;
  if (kind == SamplerKind::kDataSpace) {
    q = static_cast<float*>(arena.alloc_stack(m.dataspace.q.size() * sizeof(float)));
    MI_CUDA_CHECK(cudaMemcpyAsync(q, m.dataspace.q.data(), m.dataspace.q.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    MI_CUDA_CHECK(cudaMemcpyAsync(aux1, x.data, latent_bytes, cudaMemcpyDeviceToDevice, stream));  // z
    MI_CUDA_CHECK(cudaMemsetAsync(x.data, 0, latent_bytes, stream));                              // x0 = 0
  }

  float* xp = x.ptr<float>();
  const float s_noise = static_cast<float>(m.options.s_noise);
  int calls = 0;
  const auto call_model = [&](const float* input, int step, float sigma, float* out) {
    ModelCall c;
    c.step = step;
    c.call = calls++;
    c.sigma = sigma;
    c.timestep = kind == SamplerKind::kDataSpace ? m.dataspace.step_codes[static_cast<std::size_t>(step)]
                                                 : sigma * static_cast<float>(kNumTrainTimesteps);
    model(m.view(const_cast<float*>(input)), c, m.view(out), stream);
  };
  const auto copy_to_x = [&](const float* src) {
    MI_CUDA_CHECK(cudaMemcpyAsync(xp, src, latent_bytes, cudaMemcpyDeviceToDevice, stream));
  };

  double h_prev = 0;
  for (int i = 0; i < steps; ++i) {
    const auto si = static_cast<std::size_t>(i);
    const float s = m.sigmas[si], sn = m.sigmas[si + 1];
    const float* c = m.coef.empty() ? nullptr : m.coef.data() + si * static_cast<std::size_t>(per_step);
    switch (kind) {
      case SamplerKind::kEuler:
        call_model(xp, i, s, pred);
        if (velocity_euler) {
          // h in double is exact (difference of two fp32 values in [0, 1]); the reference rounds it to fp32.
          sk::velocity_euler(xp, pred, static_cast<float>(static_cast<double>(sn) - static_cast<double>(s)), n, stream);
        } else {
          sk::euler(xp, pred, c[sk::kEulS], c[sk::kEulDt], n, stream);
        }
        break;
      case SamplerKind::kAb2: {
        call_model(xp, i, s, pred);
        const double h = static_cast<double>(sn) - static_cast<double>(s);
        if (i == 0) {
          sk::velocity_euler(xp, pred, static_cast<float>(h), n, stream);
        } else {
          const double r = h / (2.0 * h_prev);
          sk::ab2(xp, pred, old, static_cast<float>(h), static_cast<float>(1.0 + r), static_cast<float>(r), n, stream);
        }
        h_prev = h;
        std::swap(pred, old);
        break;
      }
      case SamplerKind::kEulerAncestral:
        call_model(xp, i, s, pred);
        if (c[sk::kAncFinal] != 0.0f) {
          copy_to_x(pred);
        } else {
          const bool add_noise = m.options.eta > 0;
          if (add_noise) draw_noise(i);
          sk::ancestral(xp, pred, add_noise ? noise : nullptr, c[sk::kAncRatio], c[sk::kAncKeep], c[sk::kAncAlpha],
                        c[sk::kAncRenoise], s_noise, n, stream);
        }
        break;
      case SamplerKind::kHeun:
        call_model(xp, i, s, pred);
        if (sn == 0.0f) {
          sk::euler(xp, pred, c[sk::kHeuS], c[sk::kHeuDt], n, stream);
        } else {
          sk::heun_predict(xp, pred, c[sk::kHeuS], c[sk::kHeuDt], aux1, aux2, n, stream);
          call_model(aux2, i, sn, pred);
          sk::heun_correct(xp, aux1, aux2, pred, c[sk::kHeuSNext], c[sk::kHeuDt], n, stream);
        }
        break;
      case SamplerKind::kDpmpp2m: {
        call_model(xp, i, s, pred);
        const bool first = i == 0 || sn == 0.0f;
        sk::dpm2m(xp, pred, first ? nullptr : old, c[sk::kDpmC1], c[sk::kDpmC2], c[sk::kDpmA], c[sk::kDpmQ], n, stream);
        std::swap(pred, old);
        break;
      }
      case SamplerKind::kDpmpp2mSde:
        call_model(xp, i, s, pred);
        if (c[sk::kSdeFinal] != 0.0f) {
          copy_to_x(pred);
        } else {
          const bool add_noise = m.options.eta > 0 && m.options.s_noise > 0;
          if (add_noise) draw_noise(i);
          sk::dpm2m_sde(xp, pred, i > 0 ? old : nullptr, add_noise ? noise : nullptr, c[sk::kSdeC1], c[sk::kSdeC2],
                        c[sk::kSdeC3], c[sk::kSdeSNext], c[sk::kSdeNoise], s_noise, n, stream);
        }
        std::swap(pred, old);
        break;
      case SamplerKind::kErSde:
        call_model(xp, i, s, pred);
        if (c[sk::kErFinal] != 0.0f) {
          copy_to_x(pred);
        } else {
          const int stage = std::min(3, i + 1);
          const bool add_noise = m.options.s_noise > 0;
          if (add_noise) draw_noise(i);
          sk::er_sde(xp, pred, old, aux1, aux2, add_noise ? noise : nullptr, stage, c[sk::kErC1], c[sk::kErC2],
                     c[sk::kErS2], c[sk::kErDiv2], c[sk::kErS3], c[sk::kErDiv3], c[sk::kErAlpha], s_noise,
                     c[sk::kErNoise], n, stream);
          if (stage >= 2) std::swap(aux1, aux2);  // the new divided difference becomes the old one
        }
        std::swap(pred, old);
        break;
      case SamplerKind::kResMultistep:
        call_model(xp, i, s, pred);
        if (i == 0 || sn == 0.0f) {
          sk::euler(xp, pred, c[sk::kResS], c[sk::kResDt], n, stream);
        } else {
          sk::res_multistep(xp, pred, old, c[sk::kResExp], c[sk::kResH], c[sk::kResB1], c[sk::kResB2], n, stream);
        }
        std::swap(pred, old);
        break;
      case SamplerKind::kDataSpace: {
        const std::int64_t pixels = n / m.dataspace.channels;
        sk::dataspace_mix(aux1, xp, q, m.dataspace.channels, pixels, aux2, stream);
        call_model(aux2, i, s, pred);
        sk::dataspace_update(xp, aux1, pred, n, stream);
        break;
      }
    }
    if (after_step && !after_step(i + 1)) return i + 1;
  }
  return steps;
}

void flow_noise_scaling(const Tensor& noise, const Tensor& latent, float sigma, cudaStream_t stream) {
  check_tensor(noise, noise.numel(), "flow_noise_scaling");
  if (latent.data != nullptr) check_tensor(latent, noise.numel(), "flow_noise_scaling");
  sk::noise_scaling(noise.ptr<float>(), latent.data != nullptr ? latent.ptr<float>() : nullptr, sigma, noise.numel(), stream);
}

void velocity_to_denoised(const Tensor& x, const Tensor& v, float sigma, const Tensor& out, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check_tensor(x, n, "velocity_to_denoised");
  check_tensor(v, n, "velocity_to_denoised");
  check_tensor(out, n, "velocity_to_denoised");
  sk::velocity_to_denoised(x.ptr<float>(), v.ptr<float>(), sigma, out.ptr<float>(), n, stream);
}

void cfg_denoised(const Tensor& x, const Tensor& v_cond, const Tensor& v_uncond, float sigma, float g, const Tensor& out,
                  cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check_tensor(x, n, "cfg_denoised");
  check_tensor(v_cond, n, "cfg_denoised");
  check_tensor(v_uncond, n, "cfg_denoised");
  check_tensor(out, n, "cfg_denoised");
  sk::cfg_denoised(x.ptr<float>(), v_cond.ptr<float>(), v_uncond.ptr<float>(), sigma, g, out.ptr<float>(), n, stream);
}

void slg_denoised(const Tensor& x, const Tensor& v_cond, const Tensor& v_skip, float sigma, float scale,
                  const Tensor& out, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check_tensor(x, n, "slg_denoised");
  check_tensor(v_cond, n, "slg_denoised");
  check_tensor(v_skip, n, "slg_denoised");
  check_tensor(out, n, "slg_denoised");
  sk::slg_denoised(x.ptr<float>(), v_cond.ptr<float>(), v_skip.ptr<float>(), sigma, scale, out.ptr<float>(), n, stream);
}

void cfg_combine(const Tensor& cond, const Tensor& uncond, float g, const Tensor& out, cudaStream_t stream) {
  const std::int64_t n = cond.numel();
  check_tensor(cond, n, "cfg_combine");
  check_tensor(uncond, n, "cfg_combine");
  check_tensor(out, n, "cfg_combine");
  sk::cfg_velocity(cond.ptr<float>(), uncond.ptr<float>(), g, out.ptr<float>(), n, stream);
}

void slg_correct(const Tensor& pred, const Tensor& cond, const Tensor& skipped, float scale, const Tensor& out,
                 cudaStream_t stream) {
  const std::int64_t n = pred.numel();
  check_tensor(pred, n, "slg_correct");
  check_tensor(cond, n, "slg_correct");
  check_tensor(skipped, n, "slg_correct");
  check_tensor(out, n, "slg_correct");
  sk::slg_velocity(pred.ptr<float>(), cond.ptr<float>(), skipped.ptr<float>(), scale, out.ptr<float>(), n, stream);
}

void guide_prediction(Prediction space, const Tensor& x, float sigma, const Tensor& v_cond, const Tensor& v_uncond,
                      const Tensor& v_skip, bool cfg, float g, bool slg, float slg_scale, const Tensor& out,
                      cudaStream_t stream) {
  if (space == Prediction::kVelocity) {
    // P:416-424.
    if (cfg) cfg_combine(v_cond, v_uncond, g, out, stream);
    if (slg) slg_correct(cfg ? out : v_skip, v_cond, v_skip, slg_scale, out, stream);
    if (!cfg && !slg) {
      check_tensor(v_cond, out.numel(), "guide_prediction");
      check_tensor(out, v_cond.numel(), "guide_prediction");
      MI_CUDA_CHECK(cudaMemcpyAsync(out.data, v_cond.data, v_cond.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }
    return;
  }
  if (cfg) cfg_denoised(x, v_cond, v_uncond, sigma, g, out, stream);
  else velocity_to_denoised(x, v_cond, sigma, out, stream);
  if (slg) slg_denoised(x, v_cond, v_skip, sigma, slg_scale, out, stream);
}

void euler_update(const Tensor& x, const Tensor& v, float h, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check_tensor(x, n, "euler_update");
  check_tensor(v, n, "euler_update");
  sk::velocity_euler(x.ptr<float>(), v.ptr<float>(), h, n, stream);
}

void ab2_update(const Tensor& x, const Tensor& v, const Tensor& v_prev, float h, float a, float b, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check_tensor(x, n, "ab2_update");
  check_tensor(v, n, "ab2_update");
  check_tensor(v_prev, n, "ab2_update");
  sk::ab2(x.ptr<float>(), v.ptr<float>(), v_prev.ptr<float>(), h, a, b, n, stream);
}

}  // namespace mi::pipeline
