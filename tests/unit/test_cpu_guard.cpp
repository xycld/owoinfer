// Label guard (cmake/modules/test_labels.cmake): ctest runs every gtest case in its own process, and gives cpu-labelled
// cases MOSSINFER_EXPECT_NO_GPU=1. This global environment then fails the process if the test left a CUDA context
// behind, so a GPU test mislabelled cpu fails on GPU machines as well as in CI (where CUDA_VISIBLE_DEVICES="" makes
// every device call fail). Without a driver or a visible device there is nothing to check.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "cuda_context_check.h"

namespace {

class NoGpuGuard : public ::testing::Environment {
 public:
  void TearDown() override {
    const char* expect = std::getenv("MOSSINFER_EXPECT_NO_GPU");
    if (expect == nullptr || std::strcmp(expect, "1") != 0) return;
    const std::string report = mi::testing::cuda_context_report();
    EXPECT_TRUE(report.empty()) << "a test labelled cpu used the GPU (" << report
                                << "): give its suite the Gpu prefix or list it in cmake/modules/test_labels.cmake";
  }
};

[[maybe_unused]] ::testing::Environment* const kGuard = ::testing::AddGlobalTestEnvironment(new NoGpuGuard);

}  // namespace
