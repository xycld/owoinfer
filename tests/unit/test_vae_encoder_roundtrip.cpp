// Round trip through the bundle's VAE: image -> Flux2VaeEncoder (mode) -> Flux2VaeDecoder (DecodeMode::reference())
// -> image, PSNR against the input. Images: the tracked 256x192 crop and, when the local references exist, the
// 1024^2, 1248x1664 and 1456^2 cases. MI_VAE_ENCODER_DUMP=<dir> writes the decoded images as PPM (for LPIPS).
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae_encoder/flux2_vae_encoder.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/npy.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }

double psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double se = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    se += d * d;
  }
  return 10.0 * std::log10(255.0 * 255.0 / (se / static_cast<double>(a.size())));
}

TEST(VaeEncoderRoundTrip, EncodeDecodePsnr) {
  const fs::path vae = bundle_dir() / "vae" / "diffusion_pytorch_model.safetensors";
  if (!fs::is_regular_file(vae) || !fs::is_regular_file(bundle_dir() / "moss.mossidx")) GTEST_SKIP() << "weights not present";
  const auto index = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  mi::runtime::StreamSet streams;
  const std::size_t work = std::size_t{3} << 30;
  mi::runtime::DeviceArena arena(mi::flux2_vae_encoder::Flux2VaeEncoder::persistent_bytes(vae) +
                                 mi::flux2_vae::Flux2VaeDecoder::persistent_bytes(index) + work);
  mi::flux2_vae_encoder::Flux2VaeEncoder enc(vae, arena, streams);
  mi::flux2_vae::Flux2VaeDecoder dec(index, arena, streams);
  const cudaStream_t s = streams.compute();
  const char* dump = std::getenv("MI_VAE_ENCODER_DUMP");

  struct Case {
    std::string name;
    fs::path rgb;
    double min_psnr;
  };
  const fs::path large = source_dir() / "tools" / "reference" / "work" / "vae_encoder";
  // Measured 43.0-46.2 dB (the FLUX.2 VAE reconstructs these images closely); floors 5 dB under. A broken latent
  // layout or normalisation lands far below 20 dB.
  const Case cases[] = {{"small_192x256", source_dir() / "tests" / "data" / "vae_encoder" / "small.rgb.npy", 38.0},
                        {"fridge_1024", large / "fridge_1024.rgb.npy", 38.0},
                        {"crop_1248x1664", large / "crop_1248x1664.rgb.npy", 38.0},
                        {"fridge_1456", large / "fridge_1456.rgb.npy", 38.0}};
  for (const Case& c : cases) {
    if (!fs::is_regular_file(c.rgb)) continue;
    const auto a = mi::npy::load(c.rgb);
    const std::vector<std::uint8_t> in = a.to_vector<std::uint8_t>();
    const std::int64_t h = a.shape[0], w = a.shape[1];
    mi::DeviceBuffer lat_buf(static_cast<std::size_t>(128 * (h / 16) * (w / 16)) * 4);
    const Tensor lat = Tensor::view(lat_buf.get(), mi::DType::kF32, mi::Device::kCUDA, {128, h / 16, w / 16});
    const auto plan = enc.encode_within(Tensor::view(const_cast<std::uint8_t*>(in.data()), mi::DType::kU8, mi::Device::kCPU, {h, w, 3}),
                                        lat, {}, arena.available(), s);
    std::vector<std::uint8_t> out(in.size());
    dec.decode_within(lat, Tensor::view(out.data(), mi::DType::kU8, mi::Device::kCPU, {h, w, 3}),
                      mi::flux2_vae::DecodeMode::reference(), arena.available(), s);
    const double p = psnr(in, out);
    std::printf("[vae_encoder roundtrip] %-16s %4lldx%-4lld PSNR %.2f dB (encode %s)\n", c.name.c_str(),
                static_cast<long long>(w), static_cast<long long>(h), p, plan.tiled() ? "banded" : "untiled");
    EXPECT_GE(p, c.min_psnr) << c.name;
    if (dump != nullptr && *dump != '\0') {
      std::FILE* f = std::fopen((fs::path(dump) / (c.name + ".roundtrip.ppm")).string().c_str(), "wb");
      if (f != nullptr) {
        std::fprintf(f, "P6\n%lld %lld\n255\n", static_cast<long long>(w), static_cast<long long>(h));
        std::fwrite(out.data(), 1, out.size(), f);
        std::fclose(f);
      }
    }
  }
}

}  // namespace
