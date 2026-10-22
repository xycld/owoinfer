// pipeline::schedule_steps: the denoise steps a request runs (img2img crop, hires second pass, dataspace). CPU only.
#include <gtest/gtest.h>

#include <memory>

#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"

namespace {

namespace pl = mi::pipeline;

pl::Request base_request() {
  pl::Request r;
  r.prompt = "p";
  r.steps = 8;
  return r;
}

TEST(PipelineSchedule, Txt2img) {
  const pl::Request r = base_request();
  EXPECT_EQ(pl::schedule_steps(r), pl::make_schedule(r.scheduler, 8, {r.shift, r.flow_multiplier}).num_steps());
  EXPECT_EQ(pl::schedule_steps(r), 8);
}

TEST(PipelineSchedule, Img2imgCropsTheStart) {
  pl::Request r = base_request();
  r.init_image = std::make_shared<const pl::Image>();
  r.denoise_strength = 0.5;
  EXPECT_EQ(pl::schedule_steps(r), 4);
  r.denoise_strength = 1.0;
  EXPECT_EQ(pl::schedule_steps(r), 8);
  r.denoise_strength = 0.0;
  EXPECT_EQ(pl::schedule_steps(r), 0);
}

TEST(PipelineSchedule, HiresAddsTheSecondPass) {
  pl::Request r = base_request();
  r.hires.width = 2048;
  r.hires.height = 2048;
  r.hires.strength = 0.5;
  EXPECT_EQ(pl::schedule_steps(r), 8 + 4);  // hires steps 0: the request's
  r.hires.steps = 20;
  EXPECT_EQ(pl::schedule_steps(r), 8 + 10);
}

TEST(PipelineSchedule, DataspaceIsTheRequestedSteps) {
  pl::Request r = base_request();
  r.sampler = pl::SamplerKind::kDataSpace;
  r.steps = 4;
  EXPECT_EQ(pl::schedule_steps(r), 4);
}

}  // namespace
