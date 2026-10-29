// pipeline/pipeline_impl.h crop_workspace_bound: the forward workspace a run with crop calls (MI_RUN_CROPS) reserves.
// The bound is only a bound if no crop of the run's latent grid, with any subset of its rows, needs more under the
// installed fused route (a crop's smaller token count can move a GEMM to the op-by-op route): swept here over every crop
// size at 512^2 and 1024^2 for 2 and 3 rows, exact and q8 attention. Needs the GPU for the fused-GEMM self-check.
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline_impl.h"
#include "runtime/stream_set.h"

namespace {

namespace md = mi::moss_dit;

TEST(GpuCropBound, NoCropOrRowSubsetExceedsTheBound) {
  mi::runtime::StreamSet streams;
  const md::FusedCheck check = md::check_fused_gemms(streams.compute());
  if (!check.ok) GTEST_SKIP() << "fused GEMMs failed their self-check here: " << check.detail;
  md::FusedRouteSnapshot route;
  route.validated = check.validated;
  const std::vector<std::vector<std::int64_t>> row_sets = {{512, 26}, {77, 13, 77}, {4096, 300, 4096}};
  for (const md::AttentionMode attention : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    const md::ComputeConfig config = md::MossDit::compute_config(nullptr, attention);
    for (const std::int64_t side : {32, 64}) {
      for (const std::vector<std::int64_t>& rows : row_sets) {
        const std::size_t bound =
            mi::pipeline::crop_workspace_bound(side, side, rows, md::BlockPath::kFused, config, route);
        for (unsigned mask = 1; mask < (1u << rows.size()); ++mask) {
          std::vector<std::int64_t> subset;
          for (std::size_t k = 0; k < rows.size(); ++k) {
            if (mask & (1u << k)) subset.push_back(rows[k]);
          }
          for (std::int64_t h = 1; h <= side; ++h) {
            for (std::int64_t w = 1; w <= side; ++w) {
              const std::size_t need = md::MossDit::workspace_bytes(h, w, subset, md::BlockPath::kFused, config, &route);
              ASSERT_LE(need, bound) << "crop " << h << "x" << w << " of " << side << "x" << side << ", row mask " << mask
                                     << ", rows " << rows.size() << ", attention " << static_cast<int>(attention);
            }
          }
        }
      }
    }
  }
}

}  // namespace
