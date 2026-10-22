// pipeline/report_json: the generation report shared by `moss-infer generate --json-report` and the C API. Synthetic
// results only (host-only): valid JSON with every field, the head/tail split around the image path, U+FFFD for
// ill-formed UTF-8, null for non-finite numbers, and numbers independent of the global C++ locale.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <locale>
#include <string>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "util/json.h"

namespace {

using mi::pipeline::json_escape;
using mi::pipeline::json_number;

mi::pipeline::Request sample_request() {
  mi::pipeline::Request q;
  q.model = "/models/MOSS-Image/base";
  q.prompt = "a red apple";
  q.width = q.height = 256;
  q.steps = 2;
  q.slg_layers = {7, 9};
  return q;
}

mi::pipeline::Result sample_result() {
  mi::pipeline::Result r;
  r.seed = 1234567;
  r.width = r.height = 256;
  r.steps_scheduled = r.steps_run = 2;
  r.block_path = mi::moss_dit::BlockPath::kOpByOp;
  r.block_path_note = "fused self-check failed";
  r.timings.total_s = 12.5;
  r.timings.step_s = {0.25, 0.5};
  r.memory.arena_capacity = std::size_t{3} << 29;  // 1.5 GiB
  r.hardware.h2d_gbps = 24.0;
  r.dit_stream.streamed = true;
  r.dit_stream.resident_blocks = 20;
  r.dit_stream.buffers = 3;
  r.dit_stream.source = "host_pinned";
  r.dit_stream.bytes = 12345678901ull;
  r.dit_stream.copy_s = 0.5;
  r.dit_stream.stall_s = 0.125;
  r.dit_stream.stalls = 4;
  r.dit_stream.steps = {{20.0, 0.75, 0.0625, 2}, {24.0, 1.0, 0.0, 0}};
  return r;
}

mi::json::Value parse_fields(const std::string& fields) { return mi::json::parse("{\n" + fields + "\n}"); }

TEST(PipelineReport, FieldsAreValidJsonWithEveryKey) {
  const mi::json::Value doc =
      parse_fields(mi::pipeline::report_fields(sample_request(), sample_result(), "out.png", "lat.npy"));
  for (const char* key : {"request", "seed_used", "steps_scheduled", "steps_run", "cancelled", "cond_tokens", "uncond_tokens",
                          "image", "latents", "mode", "dit_blocks", "plan", "embedding_cache", "decode", "timings_s",
                          "memory_gib", "host_memory_gib", "hardware", "dit_stream"}) {
    EXPECT_NE(doc.find(key), nullptr) << key;
  }
  EXPECT_EQ(doc.at("image").as_string(), "out.png");
  EXPECT_EQ(doc.at("latents").as_string(), "lat.npy");
  EXPECT_EQ(doc.at("request").at("slg_layers").as_array().size(), 2u);
  EXPECT_EQ(doc.at("dit_blocks").at("path").as_string(), "op_by_op");
  EXPECT_EQ(doc.at("dit_blocks").at("note").as_string(), "fused self-check failed");
  const mi::json::Value& ds = doc.at("dit_stream");
  EXPECT_TRUE(ds.at("streamed").as_bool());
  EXPECT_EQ(ds.at("resident_blocks").as_int(), 20);
  EXPECT_EQ(ds.at("buffers").as_int(), 3);
  EXPECT_EQ(ds.at("source").as_string(), "host_pinned");
  EXPECT_EQ(ds.at("bytes").as_int(), 12345678901);
  EXPECT_EQ(ds.at("stalls").as_int(), 4);
  EXPECT_EQ(ds.at("hidden").as_double(), 0.75);  // 1 - 0.125 / 0.5
  ASSERT_EQ(ds.at("steps").as_array().size(), 2u);
  EXPECT_EQ(ds.at("steps").as_array()[0].at("stall_s").as_double(), 0.0625);
  EXPECT_EQ(ds.at("steps").as_array()[1].at("h2d_gbps").as_double(), 24.0);
  EXPECT_EQ(doc.at("timings_s").at("steps").as_array().size(), 2u);
}

// head ends with `"image": ` and tail starts with `, "latents": `, so a caller fills in the image later.
TEST(PipelineReport, PartsSplitAroundTheImage) {
  const auto q = sample_request();
  const auto r = sample_result();
  const mi::pipeline::ReportParts parts = mi::pipeline::report_parts(q, r, "lat.npy");
  const std::string image_key = "\"image\": ";
  ASSERT_GE(parts.head.size(), image_key.size());
  EXPECT_EQ(parts.head.substr(parts.head.size() - image_key.size()), image_key);
  EXPECT_EQ(parts.tail.rfind(", \"latents\": \"lat.npy\"", 0), 0u) << parts.tail.substr(0, 40);
  EXPECT_EQ(parts.head + json_escape("x.png") + parts.tail, mi::pipeline::report_fields(q, r, "x.png", "lat.npy"));
}

// Ill-formed UTF-8 becomes U+FFFD per bad byte (Unicode Table 3-7); well-formed sequences pass through.
TEST(PipelineReport, EscapeReplacesIllFormedUtf8) {
  EXPECT_EQ(json_escape("caf\xc3\xa9 \xf0\x9f\x8d\x8e"), "\"caf\xc3\xa9 \xf0\x9f\x8d\x8e\"");
  EXPECT_EQ(json_escape("a\xff" "b"), "\"a\\ufffdb\"");
  EXPECT_EQ(json_escape("\xc0\xaf"), "\"\\ufffd\\ufffd\"");          // overlong '/'
  EXPECT_EQ(json_escape("\xed\xa0\x80"), "\"\\ufffd\\ufffd\\ufffd\"");  // surrogate U+D800
  EXPECT_EQ(json_escape("\xf4\x90\x80\x80"), "\"\\ufffd\\ufffd\\ufffd\\ufffd\"");  // above U+10FFFF
  EXPECT_EQ(json_escape("\xe2\x82"), "\"\\ufffd\\ufffd\"");          // truncated
  EXPECT_EQ(json_escape("\"\\\n\r\t\x01"), "\"\\\"\\\\\\n\\r\\t\\u0001\"");
  // The report stays parseable with an ill-formed prompt, image path and note.
  auto q = sample_request();
  q.prompt = "bad\xff prompt";
  auto r = sample_result();
  r.block_path_note = "\xc3";
  const mi::json::Value doc = parse_fields(mi::pipeline::report_fields(q, r, "/tmp/out\xe9.png", ""));
  EXPECT_EQ(doc.at("request").at("prompt").as_string(), "bad\xef\xbf\xbd prompt");
  EXPECT_EQ(doc.at("image").as_string(), "/tmp/out\xef\xbf\xbd.png");
  EXPECT_EQ(doc.at("dit_blocks").at("note").as_string(), "\xef\xbf\xbd");
}

TEST(PipelineReport, NonFiniteNumbersAreNull) {
  EXPECT_EQ(json_number(std::numeric_limits<double>::quiet_NaN()), "null");
  EXPECT_EQ(json_number(std::numeric_limits<double>::infinity()), "null");
  EXPECT_EQ(json_number(-std::numeric_limits<double>::infinity()), "null");
  // %.6g spelling.
  EXPECT_EQ(json_number(1.5), "1.5");
  EXPECT_EQ(json_number(0.1234567), "0.123457");
  EXPECT_EQ(json_number(1e-7), "1e-07");
  EXPECT_EQ(json_number(1234567.0), "1.23457e+06");
  EXPECT_EQ(json_number(-0.0), "-0");
  auto r = sample_result();
  r.timings.denoise_s = std::numeric_limits<double>::quiet_NaN();
  r.timings.step_s = {std::numeric_limits<double>::infinity()};
  r.dit_stream.steps[0].gbps = std::numeric_limits<double>::quiet_NaN();
  const mi::json::Value doc = parse_fields(mi::pipeline::report_fields(sample_request(), r, "", ""));
  EXPECT_TRUE(doc.at("timings_s").at("denoise").is_null());
  EXPECT_TRUE(doc.at("timings_s").at("steps").as_array().at(0).is_null());
  EXPECT_TRUE(doc.at("dit_stream").at("steps").as_array().at(0).at("h2d_gbps").is_null());
  EXPECT_EQ(doc.at("timings_s").at("total").as_double(), 12.5);
}

// A host application's global C++ locale (comma decimal, digit grouping) never reaches the report: the stream uses
// the classic locale and the doubles are written with to_chars.
TEST(PipelineReport, IgnoresTheGlobalLocale) {
  struct CommaNumpunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
  };
  const auto q = sample_request();
  const auto r = sample_result();
  const std::string classic = mi::pipeline::report_fields(q, r, "out.png", "");
  const std::locale old = std::locale::global(std::locale(std::locale::classic(), new CommaNumpunct));
  const std::string comma = mi::pipeline::report_fields(q, r, "out.png", "");
  std::locale::global(old);
  EXPECT_EQ(comma, classic);
  const mi::json::Value doc = parse_fields(comma);
  EXPECT_EQ(doc.at("seed_used").as_int(), 1234567);
  EXPECT_EQ(doc.at("dit_stream").at("bytes").as_int(), 12345678901);
  EXPECT_EQ(doc.at("memory_gib").at("arena_capacity").as_double(), 1.5);
  EXPECT_EQ(doc.at("timings_s").at("steps").as_array().at(0).as_double(), 0.25);
}

}  // namespace
