// The DiT rows of one model call under mi_generate's guidance layout, as a pure function: the rule denoise()'s ModelFn
// applies inline (pipeline/sampler.cpp), written out for callers that evaluate one call at a time (the component API's
// MOSS guidance mode). denoise() keeps its own copy of the rule; tests check the two agree.
//   cond    always
//   uncond  while cfg_active(step), with guidance.uncond_skip while skip_active(step)
//   SLG     the cond prompt with guidance.slg_layers, while slg_active(step) and CFG is not truncated at this step
#pragma once

#include <cstdint>

#include "pipeline/sampler.h"

namespace mi::pipeline {

// Row bits, in the forward's row order (the component API's mi_run_info.rows: bit 2 = the SLG row).
inline constexpr std::uint32_t kRowCond = 1u << 0;
inline constexpr std::uint32_t kRowUncond = 1u << 1;
inline constexpr std::uint32_t kRowSlg = 1u << 2;

struct RowSelection {
  std::uint32_t rows = kRowCond;  // kRow* bits of the rows the call evaluates
  bool uncond_skip = false;       // the uncond row skips guidance.uncond_skip (else no blocks)
  // guide_prediction()'s cfg and slg flags for this call.
  bool cfg() const { return (rows & kRowUncond) != 0; }
  bool slg() const { return (rows & kRowSlg) != 0; }
};

// The rows of the call at 0-based `step` of a `num_steps`-step pass (heun's corrector call passes its step).
inline RowSelection select_rows(const Guidance& guidance, int step, int num_steps) {
  const bool skip = guidance.skip_active(step, num_steps);
  const bool cfg = guidance.cfg_active(step, num_steps);
  const bool slg_enabled = guidance.slg_layers.any() && guidance.slg_scale > 0;
  RowSelection s;
  if (cfg) {
    s.rows |= kRowUncond;
    s.uncond_skip = skip;
  }
  // CFG truncation drops the SLG row with the uncond row (cfg == do_cfg() fails only when truncated).
  if (slg_enabled && skip && cfg == guidance.do_cfg()) s.rows |= kRowSlg;
  return s;
}

}  // namespace mi::pipeline
