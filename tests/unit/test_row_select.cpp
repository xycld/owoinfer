// pipeline/row_select.h: select_rows() against tables of denoise()'s per-call row rule (pipeline/sampler.cpp ModelFn):
// CFG on and off, CFG truncation, the skip and SLG windows (start inclusive, stop exclusive, i / num_steps in double),
// and SLG with truncation. The GPU comparison with denoise()'s actual forwards belongs to the component API's tests.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "pipeline/row_select.h"

namespace {

using mi::pipeline::Guidance;
using mi::pipeline::RowSelection;
using mi::pipeline::select_rows;

// One letter per evaluated row: c = cond, u = uncond without skips, U = uncond with uncond_skip, s = the SLG row.
std::string code(const RowSelection& r) {
  std::string s;
  if (r.rows & mi::pipeline::kRowCond) s += 'c';
  if (r.rows & mi::pipeline::kRowUncond) s += r.uncond_skip ? 'U' : 'u';
  if (r.rows & mi::pipeline::kRowSlg) s += 's';
  return s;
}

void expect_table(const Guidance& g, const std::vector<std::string>& want) {
  const int n = static_cast<int>(want.size());
  for (int i = 0; i < n; ++i) {
    const RowSelection r = select_rows(g, i, n);
    EXPECT_EQ(code(r), want[static_cast<std::size_t>(i)]) << "step " << i << " of " << n;
    EXPECT_EQ(r.cfg(), (r.rows & mi::pipeline::kRowUncond) != 0);
    EXPECT_EQ(r.slg(), (r.rows & mi::pipeline::kRowSlg) != 0);
    EXPECT_FALSE(r.uncond_skip && !r.cfg()) << "step " << i;
  }
}

// The pipeline's defaults: cfg 3, the uncond row skipping block 22 over the whole pass, no SLG.
Guidance official() {
  Guidance g;
  g.cfg = 3.0;
  g.uncond_skip.set(22);
  return g;
}

Guidance with_slg() {
  Guidance g = official();
  g.slg_layers.set(7);
  g.slg_layers.set(8);
  g.slg_scale = 2.8;
  g.slg_start = 0.2;  // steps 2..5 of 10: 0.2 <= i / 10 < 0.6
  g.slg_stop = 0.6;
  return g;
}

TEST(RowSelect, OfficialDefaultsEvaluateCondAndSkippedUncondEveryStep) {
  expect_table(official(), {"cU", "cU", "cU", "cU"});
}

TEST(RowSelect, CfgOneIsTheCondRowAlone) {
  Guidance g = official();
  g.cfg = 1.0;
  expect_table(g, {"c", "c", "c"});
}

TEST(RowSelect, SkipWindowGatesTheUncondSkipSet) {
  Guidance g = official();
  g.slg_start = 0.25;  // steps 1, 2 of 4 (0.25 inclusive, 0.75 exclusive)
  g.slg_stop = 0.75;
  expect_table(g, {"cu", "cU", "cU", "cu"});
}

TEST(RowSelect, SlgRowInsideItsWindowOnly) {
  expect_table(with_slg(), {"cu", "cu", "cUs", "cUs", "cUs", "cUs", "cu", "cu", "cu", "cu"});
}

TEST(RowSelect, SlgNeedsLayersAndAPositiveScale) {
  Guidance g = with_slg();
  g.slg_scale = 0.0;
  expect_table(g, {"cu", "cu", "cU", "cU", "cU", "cU", "cu", "cu", "cu", "cu"});
  g = with_slg();
  g.slg_layers = {};
  expect_table(g, {"cu", "cu", "cU", "cU", "cU", "cU", "cu", "cu", "cu", "cu"});
}

TEST(RowSelect, CfgTruncationDropsUncondAndSlgFromItsStep) {
  Guidance g = with_slg();
  g.cfg_truncation = 0.8;  // steps 8, 9 of 10 (i / 10 >= 0.8) run the cond row alone
  expect_table(g, {"cu", "cu", "cUs", "cUs", "cUs", "cUs", "cu", "cu", "c", "c"});
  g.cfg_truncation = 0.4;  // truncated inside the SLG window: steps 4, 5 lose their SLG row too
  expect_table(g, {"cu", "cu", "cUs", "cUs", "c", "c", "c", "c", "c", "c"});
  g.cfg_truncation = 0.0;
  expect_table(g, {"c", "c", "c", "c", "c", "c", "c", "c", "c", "c"});
  g.cfg_truncation = 1.0;  // off
  expect_table(g, {"cu", "cu", "cUs", "cUs", "cUs", "cUs", "cu", "cu", "cu", "cu"});
}

TEST(RowSelect, SlgWithoutCfgIgnoresTruncation) {
  // No uncond row: truncation has nothing to drop, so the SLG row follows its window alone (cfg_active == do_cfg).
  Guidance g = with_slg();
  g.cfg = 1.0;
  g.cfg_truncation = 0.4;
  expect_table(g, {"c", "c", "cs", "cs", "cs", "cs", "c", "c", "c", "c"});
}

TEST(RowSelect, WindowPositionsAreDoubleDivisions) {
  // 1 / 3 as a double: a window starting at 1.0 / 3 includes step 1 of 3, one starting a hair above excludes it.
  Guidance g = with_slg();
  g.slg_start = 1.0 / 3.0;
  g.slg_stop = 1.0;
  expect_table(g, {"cu", "cUs", "cUs"});
  g.slg_start = 1.0 / 3.0 + 1e-16;
  expect_table(g, {"cu", "cu", "cUs"});
}

TEST(RowSelect, AStepPastTheScheduleKeepsTheRule) {
  // A corrector passes its own step; the rule only depends on (step, num_steps).
  const Guidance g = with_slg();
  EXPECT_EQ(code(select_rows(g, 3, 10)), "cUs");
  EXPECT_EQ(code(select_rows(g, 9, 10)), "cu");
}

}  // namespace
