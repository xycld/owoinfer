// The GEMM behind every linear: C[M,N] = A[M,K] * B[N,K]^T, with B in the engine's row-major weight layout, f32
// accumulation and f32 output. The fragments cannot run everywhere, so a plain compute path takes every other device.
#pragma once

#include <cstdint>
#include <memory>

#include "runtime.h"

namespace mi::vk {

enum class GemmPath { kCoop, kPlain };
enum class InputType { kF16, kBF16 };

const char* gemm_path_name(GemmPath path);

struct GemmArgs {
  const Buffer* a = nullptr;
  const Buffer* b = nullptr;
  Buffer* c = nullptr;
  std::uint32_t m = 0, n = 0, k = 0;
  std::uint32_t lda = 0, ldb = 0, ldc = 0;  // in elements; lda and ldb are multiples of 4
};

class Gemm {
 public:
  // The cooperative-matrix path where the device has it, unless `path` names one. Throws if the device cannot run it.
  explicit Gemm(Device& device);
  Gemm(Device& device, GemmPath path);
  ~Gemm();

  GemmPath path() const { return path_; }

  // Records the GEMM and a barrier. Checks the arguments. kBF16 needs the plain path; the coop path takes kF16 only,
  // because bf16 fragments are not widely available and the fallback must not depend on them.
  void record(VkCommandBuffer cb, const GemmArgs& args, InputType input = InputType::kF16) const;

 private:
  GemmPath path_;
  std::unique_ptr<Pipeline> coop_;
  std::unique_ptr<Pipeline> plain_f16_;
  std::unique_ptr<Pipeline> plain_bf16_;
};

}  // namespace mi::vk
