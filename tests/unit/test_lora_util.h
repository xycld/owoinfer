// Shared helpers of the LoRA tests: safetensors files with real values, deterministic random data.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"

namespace mi::testing::lora {

struct StTensor {
  std::string name;
  weights::DType dtype = weights::DType::kF32;
  std::vector<std::int64_t> shape;
  std::vector<float> values;  // encoded to dtype (F32, F16, BF16, I8 by truncation)
};

inline void write_st(const std::filesystem::path& path, const std::vector<StTensor>& ts, const std::string& metadata_json = "") {
  std::string header = "{";
  if (!metadata_json.empty()) header += "\"__metadata__\":" + metadata_json + ",";
  std::vector<std::uint8_t> data;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    const StTensor& t = ts[i];
    const std::size_t begin = data.size();
    for (float v : t.values) {
      switch (t.dtype) {
        case weights::DType::kF32: {
          std::uint8_t b[4];
          std::memcpy(b, &v, 4);
          data.insert(data.end(), b, b + 4);
          break;
        }
        case weights::DType::kBF16: {
          const std::uint16_t h = quant::float_to_bf16(v);
          data.push_back(static_cast<std::uint8_t>(h & 0xFF));
          data.push_back(static_cast<std::uint8_t>(h >> 8));
          break;
        }
        case weights::DType::kF16: {
          const _Float16 f = static_cast<_Float16>(v);
          std::uint16_t h;
          std::memcpy(&h, &f, 2);
          data.push_back(static_cast<std::uint8_t>(h & 0xFF));
          data.push_back(static_cast<std::uint8_t>(h >> 8));
          break;
        }
        default: data.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(v))); break;
      }
    }
    std::string shape = "[";
    for (std::size_t d = 0; d < t.shape.size(); ++d) shape += (d ? "," : "") + std::to_string(t.shape[d]);
    shape += "]";
    header += (i ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"" + weights::dtype_name(t.dtype) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(data.size()) + "]}";
  }
  header += "}";
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const std::uint64_t len = header.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

// Values ~ N(0, scale^2), rounded to bf16 so that every dtype stores them exactly (F16 for |v| in its normal range).
inline std::vector<float> randn(std::size_t n, std::uint32_t seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = quant::bf16_to_float(quant::float_to_bf16(dist(rng)));
  return v;
}

inline std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
  std::vector<std::uint16_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = quant::float_to_bf16(v[i]);
  return out;
}

inline const char* kSft = "/workspace/Models/MOSS-Image/sft/transformer";
inline const char* kInt8File = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
inline const char* kKohyaLora = "/workspace/Models/community/congruency/MOSS-Image-Turbo-MFD/MOSS-Image-SFT-Turbo-MFD-v0.1.comfy.safetensors";
inline const char* kPeftLora = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
inline const char* kTinyLora = "/workspace/Models/community/nynxz/MOSS-Image/moss_sft_destripe_top64_bf16.safetensors";
inline const char* kLohaLora = "/workspace/Models/community/comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test/moss_hires_lora_test_v0.1.safetensors";

}  // namespace mi::testing::lora
