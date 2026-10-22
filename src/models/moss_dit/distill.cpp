#include "models/moss_dit/distill.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <system_error>

#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"

namespace mi::moss_dit {
namespace {

namespace fs = std::filesystem;
constexpr std::int64_t kDim = 3072, kFreq = 256;

[[noreturn]] void fail(const std::string& msg) { throw DistillError("mi::moss_dit: " + msg); }

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const int exp = (h >> 10) & 0x1f;
  const std::uint32_t man = h & 0x3ffu;
  float v = 0;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    v = man != 0 ? std::nanf("") : INFINITY;
  } else {
    v = std::ldexp(static_cast<float>(man | 0x400u), exp - 25);
  }
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  u |= sign;
  std::memcpy(&v, &u, 4);
  return v;
}

std::vector<float> read_floats(const fs::path& path, const weights::TensorInfo& t) {
  const std::size_t es = t.dtype == weights::DType::kF32 ? 4 : 2;
  std::vector<std::byte> raw(static_cast<std::size_t>(t.nbytes));
  weights::FileReader(path).read_at(t.offset, raw.data(), raw.size());
  std::vector<float> out(raw.size() / es);
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (t.dtype == weights::DType::kF32) {
      std::memcpy(&out[i], raw.data() + 4 * i, 4);
    } else {
      std::uint16_t h = 0;
      std::memcpy(&h, raw.data() + 2 * i, 2);
      out[i] = t.dtype == weights::DType::kBF16 ? quant::bf16_to_float(h) : half_to_float(h);
    }
    if (!std::isfinite(out[i])) fail(util::path_utf8(path) + ": '" + t.name + "' holds a non-finite value");
  }
  return out;
}

}  // namespace

DistillBundle open_distill_bundle(const fs::path& dir) {
  // error_code overloads: an unreadable path (permissions, a broken mount) is a DistillError, not a filesystem_error.
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    fail("distill bundle not found (not a directory): " + util::path_utf8(dir) + (ec ? " (" + ec.message() + ")" : ""));
  }
  DistillBundle b;
  b.dir = dir;
  const auto pick = [&](const char* name) {
    std::error_code e;
    const bool file = fs::is_regular_file(dir / name, e);
    if (e && e != std::errc::no_such_file_or_directory) fail(util::path_utf8(dir / name) + ": " + e.message());
    return file ? dir / name : fs::path();
  };
  b.time_embedding = pick("time_embedding.safetensors");
  b.lora = pick("lora.safetensors");
  b.dataspace = pick("dataspace.safetensors");
  if (b.time_embedding.empty() && b.lora.empty() && b.dataspace.empty()) {
    fail(util::path_utf8(dir) + " holds none of time_embedding.safetensors, lora.safetensors, dataspace.safetensors");
  }
  return b;
}

TimeEmbeddingFp32 load_time_embedding(const fs::path& file) {
  try {
    const weights::SafetensorsFile st(file);
    struct Want {
      std::vector<float>* dst;
      std::vector<std::int64_t> shape;
      bool required;
    };
    TimeEmbeddingFp32 te;
    const std::map<std::string, Want, std::less<>> want = {
        {"time_embedding.0.weight", {&te.w0, {kDim, kFreq}, true}},
        {"time_embedding.0.bias", {&te.b0, {kDim}, true}},
        {"time_embedding.2.weight", {&te.w2, {kDim, kDim}, true}},
        {"time_embedding.2.bias", {&te.b2, {kDim}, true}},
        {"shared_block_modulation.bias", {&te.mod_bias, {6 * kDim}, false}},
        {"video_head_modulation.bias", {&te.head_bias, {2 * kDim}, false}},
    };
    std::map<std::string, bool, std::less<>> seen;
    for (const weights::TensorInfo& t : st.tensors()) {
      std::string_view name = t.name;
      for (std::string_view p : {"model.diffusion_model.", "diffusion_model."}) {
        if (name.starts_with(p)) {
          name.remove_prefix(p.size());
          break;
        }
      }
      const auto it = want.find(name);
      if (it == want.end()) fail(util::path_utf8(file) + ": '" + t.name + "' is not part of a time-embedding replacement");
      if (seen[std::string(name)]) fail(util::path_utf8(file) + ": '" + std::string(name) + "' appears twice");
      seen[std::string(name)] = true;
      if (t.dtype != weights::DType::kF32 && t.dtype != weights::DType::kBF16 && t.dtype != weights::DType::kF16) {
        fail(util::path_utf8(file) + ": '" + t.name + "' must be F32, BF16 or F16");
      }
      if (t.shape != it->second.shape) fail(util::path_utf8(file) + ": '" + t.name + "' has an unexpected shape");
      *it->second.dst = read_floats(file, t);
    }
    for (const auto& [name, w] : want) {
      if (w.required && !seen[name]) fail(util::path_utf8(file) + ": '" + name + "' missing");
    }
    return te;
  } catch (const DistillError&) {
    throw;
  } catch (const std::exception& e) {
    fail(util::path_utf8(file) + ": " + e.what());
  }
}

std::vector<float> time_vector_fp32(const TimeEmbeddingFp32& te, float timestep) {
  // The official sinusoid (float64, then fp32), kept in fp32 as an fp32 embedder takes it.
  constexpr int kHalf = kFreq / 2;
  float emb[kFreq];
  for (int i = 0; i < kHalf; ++i) {
    const double freq = std::pow(10000.0, -static_cast<double>(i) / kHalf);
    const double phase = static_cast<double>(timestep) * freq;
    emb[i] = static_cast<float>(std::cos(phase));
    emb[kHalf + i] = static_cast<float>(std::sin(phase));
  }
  std::vector<float> h(kDim), time(kDim);
  for (std::int64_t r = 0; r < kDim; ++r) {
    double acc = te.b0[static_cast<std::size_t>(r)];
    const float* w = te.w0.data() + r * kFreq;
    for (std::int64_t c = 0; c < kFreq; ++c) acc += static_cast<double>(w[c]) * emb[c];
    const auto v = static_cast<float>(acc);
    h[static_cast<std::size_t>(r)] = v / (1.0f + std::exp(-v));  // silu
  }
  for (std::int64_t r = 0; r < kDim; ++r) {
    double acc = te.b2[static_cast<std::size_t>(r)];
    const float* w = te.w2.data() + r * kDim;
    for (std::int64_t c = 0; c < kDim; ++c) acc += static_cast<double>(w[c]) * h[static_cast<std::size_t>(c)];
    time[static_cast<std::size_t>(r)] = static_cast<float>(acc);
  }
  return time;
}

}  // namespace mi::moss_dit
