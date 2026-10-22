// Host-side image work of img2img, inpainting and the hires fix (roadmap P7b): images and masks in, their sizing to
// the request, the latent-grid mask, the per-step inpainting blend and the image file reader. No CUDA.
//
// Sizing follows diffusers' VaeImageProcessor on PIL images (resize_mode "default" / "crop" / "fill"), with PIL's
// Image.resize(LANCZOS) reproduced bit for bit (Resample.c: fixed-point 8-bit two-pass separable filter, horizontal
// pass first, intermediate rounded to 8 bits). The mask follows diffusers' inpainting pipelines (FLUX.2 Klein
// inpaint): sized like the image, binarized at 0.5, then bilinearly interpolated (align_corners = False, no
// antialiasing) to the latent grid [H/16, W/16].
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mi::pipeline {

// Input image bounds (init images, masks, image files): any size up to this side, this many pixels and (files) bytes.
inline constexpr int kMaxInputSide = 16384;
inline constexpr std::int64_t kMaxInputPixels = 64'000'000;
inline constexpr std::size_t kMaxImageFileBytes = std::size_t{64} << 20;

// An 8-bit image, row-major [height, width, channels]: RGB (3 channels) or grayscale (1).
struct Image {
  int width = 0, height = 0, channels = 3;
  std::vector<std::uint8_t> pixels;
  std::string name;  // optional label for reports and PNG metadata (e.g. the file's base name)
  bool empty() const { return pixels.empty(); }
};

// How an image is brought to the request size (diffusers' resize_mode on PIL images):
//   kCrop     scale to cover the target keeping the aspect ratio, then center-crop ("crop"). The default: no
//             distortion and no invented borders.
//   kFit      scale to fit inside keeping the aspect ratio, center it, and fill the two bands by stretching the edge
//             rows/columns (PIL's default bicubic resize of a zero-height box, "fill"); an inpainting mask's bands are
//             filled the same way from the mask.
//   kStretch  scale each axis independently ("default").
// All three return the image unchanged when it already has the target size.
enum class ResizeMode { kCrop, kFit, kStretch };
const char* resize_mode_name(ResizeMode mode);
// crop, fit, stretch; throws std::invalid_argument otherwise.
ResizeMode parse_resize_mode(std::string_view name);

// PIL Image.resize((width, height), Image.LANCZOS, box=(box_x0, box_y0, box_x1, box_y1)), bit-exact (PIL's
// 8-bit path, any channel count). The default box (all negative) is the whole image. Throws std::invalid_argument for
// non-positive sizes or a pixel buffer that does not match the image size.
Image resize_lanczos(const Image& src, int width, int height, float box_x0 = -1, float box_y0 = -1, float box_x1 = -1,
                     float box_y1 = -1);

// diffusers VaeImageProcessor.resize(image, height, width, resize_mode) for a PIL image (see ResizeMode).
Image resize_to(const Image& src, int width, int height, ResizeMode mode);

// PIL Image.convert("L") of an RGB image: (R * 19595 + G * 38470 + B * 7471 + 0x8000) >> 16. A 1-channel image is
// returned unchanged.
Image to_grayscale(const Image& src);

// The inpainting mask on the latent grid: `mask` (1 channel, already at the request size) binarized (value >= 128,
// i.e. >= 0.5 after /255, means regenerate) and bilinearly interpolated to [latent_h, latent_w] as torch's
// F.interpolate(mode="bilinear", align_corners=False) does on the CPU in fp32 (its generic kernel's arithmetic; at the
// pipeline's exact factor of 16 its separable kernel gives the same values: the mean of each 16 x 16 cell's 2 x 2
// central pixels). Values lie in [0, 1]: 1 = regenerate, 0 = keep the init image.
std::vector<float> latent_mask(const Image& mask, int latent_h, int latent_w);

// diffusers' per-step inpainting blend after a sampling step (all fp32, one rounding per torch op):
//   proper = sigma * noise + (1 - sigma) * z0          (scheduler.scale_noise at the next step's sigma)
//   x      = (1 - m) * proper + m * x
// x, z0, noise: [channels, h, w]; mask: [h, w], broadcast over the channels. At sigma = 0 (after the last step)
// proper is z0 itself.
void inpaint_blend(std::span<float> x, std::span<const float> z0, std::span<const float> noise, std::span<const float> mask,
                   float sigma);

// The schedule offset of a denoise strength (diffusers get_timesteps): of `num_steps` steps, the first
//   int(max(num_steps - min(num_steps * strength, num_steps), 0))
// are skipped and sampling starts at sigmas[that index]. strength 1 -> 0 (all steps), strength 0 -> num_steps (none).
int strength_start_step(int num_steps, double strength);

// Untrusted input: the decoder is the vendored stb_image (third_party/stb), hardened here by a file-size cap
// (kMaxImageFileBytes, bounded read of regular files only) and a header check before any pixel allocation (at most
// kMaxInputSide per side and kMaxInputPixels), and fuzzed (tests/fuzz). It is not a memory-safe decoder: a front end
// that decodes uploads from untrusted parties (a server) should decode them with a memory-safe decoder (e.g. Wuffs)
// or in a sandboxed process and pass the pixels in Request::init_image / mask instead.
//
// Reads a PNG or JPEG file (the vendored stb_image decoder) as RGB (channels 3: gray expanded, alpha dropped, 16-bit
// samples reduced to 8) or grayscale (channels 1: PIL's convert("L") of the RGB image). A JPEG's EXIF orientation is
// applied (PIL ImageOps.exif_transpose, as diffusers.utils.load_image does). `name` is set to the file's base name.
// Throws std::runtime_error on a missing or non-regular file, a file over kMaxImageFileBytes, an undecodable one, or an
// image over kMaxInputSide on a side or kMaxInputPixels.
Image load_image(const std::filesystem::path& file, int channels = 3);
// The same from an in-memory PNG / JPEG file.
Image decode_image(std::span<const std::uint8_t> file, int channels = 3);

}  // namespace mi::pipeline
