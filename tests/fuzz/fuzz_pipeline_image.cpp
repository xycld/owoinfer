// libFuzzer target for the untrusted-input image path (src/pipeline/pipeline_image: stb_image PNG/JPEG decoding behind
// the header and size checks, the EXIF orientation parser, grayscale conversion). Built by tests/fuzz/CMakeLists.txt
// (clang, -fsanitize=fuzzer,address,undefined), opt-in through -DMOSSINFER_FUZZ=ON:
//   build-fuzz/fuzz_pipeline_image -max_total_time=300 tests/fuzz/corpus_pipeline_image
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "pipeline/pipeline_image.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const std::span<const std::uint8_t> file(data, size);
  try {
    const mi::pipeline::Image im = mi::pipeline::decode_image(file, (data[0] & 1) ? 1 : 3);
    if (im.pixels.size() != static_cast<std::size_t>(im.width) * static_cast<std::size_t>(im.height) *
                                static_cast<std::size_t>(im.channels)) {
      __builtin_trap();
    }
  } catch (const std::runtime_error&) {
    // refused input
  }
  return 0;
}
