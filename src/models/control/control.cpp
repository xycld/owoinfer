#include "models/control/control.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_codec.h"

namespace mi::control {
namespace {

using WDType = weights::DType;

[[noreturn]] void fail(const std::filesystem::path& file, const std::string& msg) {
  throw ControlError("control adapter " + util::path_utf8(file) + ": " + msg);
}

bool is_float(WDType d) { return d == WDType::kF32 || d == WDType::kF16 || d == WDType::kBF16; }

std::string trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return std::string(s);
}

float widen(const std::byte* p, WDType d) {
  if (d == WDType::kF32) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
  }
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  if (d == WDType::kBF16) return quant::bf16_to_float(h);
  const int e = (h >> 10) & 0x1F, m = h & 0x3FF;  // F16
  const float mag = e == 0    ? std::ldexp(static_cast<float>(m), -24)
                    : e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                              : std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, e - 15);
  return (h & 0x8000) ? -mag : mag;
}

// A tensor's values as bf16 bits (BF16 copied, the others rounded); ControlError for a non-finite value.
std::vector<std::uint16_t> read_bf16(const weights::FileReader& reader, const weights::TensorInfo& t,
                                     const std::filesystem::path& file) {
  std::vector<std::byte> raw(static_cast<std::size_t>(t.nbytes));
  reader.read_at(t.offset, raw.data(), raw.size());
  const std::size_t n = static_cast<std::size_t>(t.numel()), es = weights::dtype_size(t.dtype);
  std::vector<std::uint16_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float v = widen(raw.data() + i * es, t.dtype);
    if (!std::isfinite(v)) fail(file, t.name + " has a non-finite value");
    if (t.dtype == WDType::kBF16) {
      std::memcpy(&out[i], raw.data() + i * es, 2);
    } else {
      out[i] = quant::float_to_bf16(v);
    }
  }
  return out;
}

}  // namespace

int ControlFile::kind_index(std::string_view kind) const {
  const auto it = std::find(kinds.begin(), kinds.end(), kind);
  return it == kinds.end() ? -1 : static_cast<int>(it - kinds.begin());
}

ControlFile open_control_file(const std::filesystem::path& path) {
  const weights::SafetensorsFile st(path);
  ControlFile f;
  f.path = path;
  const weights::TensorInfo* w = st.find(kEmbedderWeight);
  const weights::TensorInfo* b = st.find(kEmbedderBias);
  if (w == nullptr || b == nullptr) fail(path, std::string("no ") + kEmbedderWeight + " / " + kEmbedderBias + " (not a control adapter)");
  if (!is_float(w->dtype) || !is_float(b->dtype)) {
    fail(path, std::string("the embedder must be F32, F16 or BF16, got ") + weights::dtype_name(w->dtype) + " / " +
                   weights::dtype_name(b->dtype));
  }
  if (w->shape.size() != 2 || w->shape[0] != kDim || w->shape[1] % kGroupChannels != 0 || w->shape[1] < 2 * kGroupChannels) {
    fail(path, std::string(kEmbedderWeight) + " must be [" + std::to_string(kDim) + ", 128 * (1 + G)] with G >= 1 control latents");
  }
  if (b->shape.size() != 1 || b->shape[0] != kDim) fail(path, std::string(kEmbedderBias) + " must be [" + std::to_string(kDim) + "]");
  f.in_features = w->shape[1];
  f.dtype = w->dtype;
  const std::int64_t groups = f.in_features / kGroupChannels - 1;
  const auto& meta = st.metadata();
  if (const auto it = meta.find("control_channels"); it != meta.end()) {
    std::string_view s = it->second;
    for (std::size_t pos = 0;;) {
      const std::size_t comma = s.find(',', pos);
      f.kinds.push_back(trim(s.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos)));
      if (comma == std::string_view::npos) break;
      pos = comma + 1;
    }
    bool ok = static_cast<std::int64_t>(f.kinds.size()) == groups;
    for (std::size_t i = 0; ok && i < f.kinds.size(); ++i) {
      ok = !f.kinds[i].empty() && std::count(f.kinds.begin(), f.kinds.end(), f.kinds[i]) == 1;
    }
    if (!ok) {
      fail(path, "metadata control_channels \"" + it->second + "\" must name " + std::to_string(groups) +
                     " distinct kinds (the embedder has " + std::to_string(groups) + " control latents)");
    }
  } else {
    for (std::int64_t g = 0; g < groups; ++g) f.kinds.push_back("control" + std::to_string(g));
  }
  if (const auto it = meta.find("base_model"); it != meta.end()) f.base_model = it->second;
  return f;
}

bool is_control_file(const std::filesystem::path& path) {
  try {
    const weights::SafetensorsFile st(path);
    const weights::TensorInfo* w = st.find(kEmbedderWeight);
    return w != nullptr && w->shape.size() == 2 && w->shape[1] > kGroupChannels;
  } catch (const std::exception&) {
    return false;
  }
}

EmbedderWeights read_embedder(const ControlFile& file) {
  const weights::SafetensorsFile st(file.path);
  const weights::TensorInfo* w = st.find(kEmbedderWeight);
  const weights::TensorInfo* b = st.find(kEmbedderBias);
  // The file may have changed since open_control_file: check again.
  if (w == nullptr || b == nullptr || !is_float(w->dtype) || !is_float(b->dtype) || w->shape.size() != 2 ||
      w->shape[0] != kDim || w->shape[1] != file.in_features || b->shape.size() != 1 || b->shape[0] != kDim) {
    fail(file.path, "the embedder changed since the file was opened");
  }
  const weights::FileReader reader(file.path);
  EmbedderWeights e;
  e.out_features = kDim;
  e.in_features = file.in_features;
  e.w = read_bf16(reader, *w, file.path);
  e.bias = read_bf16(reader, *b, file.path);
  return e;
}

std::size_t DeviceEmbedder::device_bytes(std::int64_t out_features, std::int64_t in_features) {
  // Weight then bias, the bias at a 256-byte boundary.
  const std::size_t wb = static_cast<std::size_t>(out_features * in_features) * 2;
  return (wb + 255) / 256 * 256 + static_cast<std::size_t>(out_features) * 2;
}

DeviceEmbedder::DeviceEmbedder(const EmbedderWeights& w)
    : buf_(device_bytes(w.out_features, w.in_features)), out_(w.out_features), in_(w.in_features) {
  if (w.w.size() != static_cast<std::size_t>(out_ * in_) || w.bias.size() != static_cast<std::size_t>(out_)) {
    throw std::invalid_argument("mi::control::DeviceEmbedder: weight / bias sizes do not match");
  }
  const std::size_t wb = w.w.size() * 2;
  buf_.upload(w.w.data(), wb);
  buf_.upload(w.bias.data(), w.bias.size() * 2, (wb + 255) / 256 * 256);
}

Tensor DeviceEmbedder::weight() const { return Tensor::view(buf_.get(), DType::kBF16, Device::kCUDA, {out_, in_}); }

Tensor DeviceEmbedder::bias() const {
  const std::size_t off = (static_cast<std::size_t>(out_ * in_) * 2 + 255) / 256 * 256;
  return Tensor::view(static_cast<std::byte*>(buf_.get()) + off, DType::kBF16, Device::kCUDA, {out_});
}

bool window_active(double start, double end, int step, int num_steps) {
  if (num_steps <= 0) return false;
  const double f = static_cast<double>(step) / static_cast<double>(num_steps);
  return start <= f && f < end;
}

PassInputs::PassInputs(int groups, std::int64_t h, std::int64_t w) : groups_(static_cast<std::size_t>(std::max(groups, 0))), h_(h), w_(w) {}

void PassInputs::set(int group, std::vector<float> latents, double start, double end) {
  if (group < 0 || group >= groups()) throw std::invalid_argument("mi::control::PassInputs: no group " + std::to_string(group));
  if (latents.size() != static_cast<std::size_t>(kGroupChannels * h_ * w_)) {
    throw std::invalid_argument("mi::control::PassInputs: a control latent must be [128, " + std::to_string(h_) + ", " +
                                std::to_string(w_) + "]");
  }
  Group& g = groups_[static_cast<std::size_t>(group)];
  g.latents = std::move(latents);
  g.start = start;
  g.end = end;
  g.written = -1;
}

int PassInputs::apply(const Tensor& extra, int step, int num_steps, cudaStream_t stream) {
  const std::int64_t plane = kGroupChannels * h_ * w_;
  if (extra.device != Device::kCUDA || extra.dtype != DType::kF32 || !extra.is_contiguous() ||
      extra.numel() != plane * groups()) {
    throw std::invalid_argument("mi::control::PassInputs::apply: extra must be a contiguous CUDA f32 [" +
                                std::to_string(groups() * kGroupChannels) + ", " + std::to_string(h_) + ", " +
                                std::to_string(w_) + "], got " + extra.describe());
  }
  int written = 0;
  const std::size_t bytes = static_cast<std::size_t>(plane) * sizeof(float);
  for (int i = 0; i < groups(); ++i) {
    Group& g = groups_[static_cast<std::size_t>(i)];
    const int want = !g.latents.empty() && window_active(g.start, g.end, step, num_steps) ? 1 : 0;
    if (want == g.written) continue;
    void* dst = static_cast<std::byte*>(extra.data) + static_cast<std::size_t>(i) * bytes;
    if (want == 1) {
      MI_CUDA_CHECK(cudaMemcpyAsync(dst, g.latents.data(), bytes, cudaMemcpyHostToDevice, stream));
    } else {
      MI_CUDA_CHECK(cudaMemsetAsync(dst, 0, bytes, stream));
    }
    g.written = want;
    ++written;
  }
  return written;
}

}  // namespace mi::control
