// Whether a DiT with NVFP4 linears can run here. User decision D1 (2026-10-23): NVFP4 files are refused on GPUs
// without FP4 tensor cores, pointing to the INT8 file; there is no bf16 decode fallback. The NVFP4 GEMMs exist only as
// sm_120a SASS (kernels/nvfp4: cc exactly 12.0 and a build with those kernels).
#pragma once

#include <string>

namespace mi::moss_dit {

// Empty when NVFP4 linears can run, else the message for the user. major < 0: no CUDA device. The GPU checks come
// first, so a GPU without FP4 tensor cores always gets the D1 message.
std::string nvfp4_unsupported_reason(int major, int minor, bool kernels_available);
// The same for the current device (kernels::nvfp4::available()).
std::string nvfp4_unsupported_reason();

}  // namespace mi::moss_dit
