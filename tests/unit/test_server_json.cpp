// tools/server/json_out.h: moss-server's own JSON output helpers (it cannot link the pipeline's renderer) must write
// exactly what pipeline/report_json.h writes. Strings: every 1- and 2-byte input and the 3-byte inputs around each
// UTF-8 lead byte. Numbers: under a comma-decimal C++ locale, and a comma-decimal C locale where the system has one.
#include <gtest/gtest.h>

#include <clocale>
#include <cstdio>
#include <iterator>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

#include "../../tools/server/json_out.h"
#include "../../tools/server/names.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/accel/accel.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/report_json.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/sampler_driver.h"
#include "pipeline/samplers/schedules.h"
#include "planner/planner.h"

namespace {

TEST(ServerJson, EscapeMatchesThePipeline) {
  const auto check = [](const std::string& s) {
    ASSERT_EQ(mi::server::str(s), mi::pipeline::json_escape(s)) << "input of " << s.size() << " bytes";
  };
  check("");
  check("a \"quoted\" \\ path\twith\r\nlines \x7f and \xc3\xa9\xe2\x82\xac\xf0\x9f\x8d\x8e");
  for (int a = 0; a < 256; ++a) {
    check(std::string(1, static_cast<char>(a)));
    for (int b = 0; b < 256; ++b) check(std::string{static_cast<char>(a), static_cast<char>(b)});
  }
  for (int a = 0xc0; a < 256; ++a) {
    for (int b = 0x7f; b <= 0xc0; ++b) {
      for (const int c : {0x41, 0x80, 0xbf, 0xc0}) {
        check(std::string{static_cast<char>(a), static_cast<char>(b), static_cast<char>(c)});
      }
    }
  }
}

TEST(ServerJson, NumbersMatchThePipelineAndIgnoreTheLocale) {
  struct CommaNumpunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
  };
  const double values[] = {0.0,
                           -0.0,
                           1.5,
                           -2.25,
                           5.949237,
                           1e-7,
                           123456789.0,
                           1e21,
                           std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::max(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
  const std::locale old_cpp = std::locale::global(std::locale(std::locale::classic(), new CommaNumpunct));
  const std::string old_c = std::setlocale(LC_NUMERIC, nullptr);
  std::string c_locale = "C";
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "ru_RU.UTF-8", "German_Germany.1252"}) {
    if (std::setlocale(LC_NUMERIC, name) != nullptr) {
      c_locale = name;
      break;
    }
  }
  std::string server[std::size(values)], pipeline[std::size(values)];
  for (std::size_t i = 0; i < std::size(values); ++i) {
    server[i] = mi::server::num(values[i]);
    pipeline[i] = mi::pipeline::json_number(values[i]);
  }
  std::setlocale(LC_NUMERIC, old_c.c_str());
  std::locale::global(old_cpp);
  std::printf("[server_json] numbers written under C locale %s and a comma-decimal C++ locale\n", c_locale.c_str());

  for (std::size_t i = 0; i < std::size(values); ++i) EXPECT_EQ(server[i], pipeline[i]) << i;
  EXPECT_EQ(server[2], "1.5");
  EXPECT_EQ(server[4], "5.94924");
  EXPECT_EQ(server[5], "1e-07");
  EXPECT_EQ(server[10], "null");
  EXPECT_EQ(server[11], "null");
  EXPECT_EQ(server[12], "null");
}

// tools/server/names.h: every protocol name is the pipeline's own spelling, and maps to the C value of the pipeline
// value (the C enums keep the pipeline's order; mi_tier starts with the reserved AUTO, which has no name, mi_attention
// with DEFAULT = unset). Each C value appears once.
TEST(ServerJson, EnumNamesAreThePipelines) {
  namespace p = mi::pipeline;
  int i = 0;
  for (const auto& n : mi::server::kSamplerNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::parse_sampler(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::sampler_name(p::parse_sampler(n.name)), n.name);
  }
  EXPECT_EQ(i, 9);
  i = 0;
  for (const auto& n : mi::server::kSchedulerNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::parse_scheduler(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::scheduler_name(p::parse_scheduler(n.name)), n.name);
  }
  EXPECT_EQ(i, 7);
  i = 0;
  for (const auto& n : mi::server::kNoiseLayoutNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::parse_noise_layout(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::noise_layout_name(p::parse_noise_layout(n.name)), n.name);
  }
  EXPECT_EQ(i, 2);
  i = 0;
  for (const auto& n : mi::server::kNoiseSourceNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::parse_noise_source(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::noise_source_name(p::parse_noise_source(n.name)), n.name);
  }
  EXPECT_EQ(i, 3);
  const mi::planner::Tier tiers[] = {mi::planner::Tier::kReference, mi::planner::Tier::kHigh, mi::planner::Tier::kTurbo};
  i = 0;
  for (const auto& n : mi::server::kTierNames) {
    EXPECT_EQ(static_cast<int>(n.value), i + 1);  // after MI_TIER_AUTO
    EXPECT_EQ(mi::planner::parse_tier(n.name), tiers[i]) << n.name;
    EXPECT_EQ(mi::planner::tier_name(tiers[i++]), n.name);
  }
  EXPECT_EQ(i, 3);
  i = 0;
  for (const auto& n : mi::server::kAttentionNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    if (n.value == MI_ATTENTION_DEFAULT) {
      EXPECT_EQ(n.name, "default");
      EXPECT_THROW(mi::planner::parse_attention(n.name), std::invalid_argument);
      continue;
    }
    EXPECT_EQ(static_cast<int>(mi::planner::parse_attention(n.name)) + 1, static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(mi::planner::attention_name(mi::planner::parse_attention(n.name)), n.name);
  }
  EXPECT_EQ(i, 6);
  i = 0;
  for (const auto& n : mi::server::kResizeNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::parse_resize_mode(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::resize_mode_name(p::parse_resize_mode(n.name)), n.name);
  }
  EXPECT_EQ(i, 3);
  i = 0;
  for (const auto& n : mi::server::kMaskEncodingNames) EXPECT_EQ(static_cast<int>(n.value), i++);  // the server's own names
  EXPECT_EQ(i, 2);
  i = 0;
  for (const auto& n : mi::server::kStepCacheNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(p::accel::parse_step_cache(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(p::accel::step_cache_name(p::accel::parse_step_cache(n.name)), n.name);
  }
  EXPECT_EQ(i, 2);
  const mi::planner::TeHostPin pins[] = {mi::planner::TeHostPin::kAuto, mi::planner::TeHostPin::kOn,
                                         mi::planner::TeHostPin::kOff};
  i = 0;
  for (const auto& n : mi::server::kTeHostPinNames) {
    EXPECT_EQ(static_cast<int>(n.value), i);
    EXPECT_EQ(mi::planner::parse_te_host_pin(n.name), pins[i]) << n.name;
    EXPECT_EQ(mi::planner::te_host_pin_name(pins[i++]), n.name);
  }
  EXPECT_EQ(i, 3);
  i = 0;
  for (const auto& n : mi::server::kDitDirectIoNames) EXPECT_EQ(static_cast<int>(n.value), i++);  // the server's own names
  EXPECT_EQ(i, 3);
  i = 0;
  for (const auto& n : mi::server::kModeNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    EXPECT_EQ(static_cast<int>(mi::planner::parse_mode(n.name)), static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(mi::planner::mode_name(mi::planner::parse_mode(n.name)), n.name);
  }
  EXPECT_EQ(i, 3);
  i = 0;
  for (const auto& n : mi::server::kBlockPathNames) {
    EXPECT_EQ(static_cast<int>(n.value), i++);
    if (n.value == MI_BLOCK_PATH_DEFAULT) {
      EXPECT_EQ(n.name, "default");
      EXPECT_THROW(mi::moss_dit::parse_block_path(n.name), std::invalid_argument);
      continue;
    }
    EXPECT_EQ(static_cast<int>(mi::moss_dit::parse_block_path(n.name)) + 1, static_cast<int>(n.value)) << n.name;
    EXPECT_EQ(mi::moss_dit::block_path_name(mi::moss_dit::parse_block_path(n.name)), n.name);
  }
  EXPECT_EQ(i, 4);
}

// hello's "values": every enum table, each under the parameter it validates, with the table's names in order.
TEST(ServerJson, HelloValuesAreTheTables) {
  namespace s = mi::server;
  const auto same = [](std::span<const std::string_view> listed, const auto& table) {
    ASSERT_EQ(listed.size(), std::size(table));
    for (std::size_t i = 0; i < listed.size(); ++i) EXPECT_EQ(listed[i], table[i].name);
  };
  ASSERT_EQ(std::size(s::kValueTables), 14u);
  for (const s::ValueTable& t : s::kValueTables) {
    SCOPED_TRACE(t.param);
    if (t.param == "mode") same(t.names, s::kModeNames);
    else if (t.param == "block_path") same(t.names, s::kBlockPathNames);
    else if (t.param == "tier") same(t.names, s::kTierNames);
    else if (t.param == "attention") same(t.names, s::kAttentionNames);
    else if (t.param == "dit_direct_io") same(t.names, s::kDitDirectIoNames);
    else if (t.param == "te_host_pin") same(t.names, s::kTeHostPinNames);
    else if (t.param == "sampler") same(t.names, s::kSamplerNames);
    else if (t.param == "scheduler") same(t.names, s::kSchedulerNames);
    else if (t.param == "noise_layout") same(t.names, s::kNoiseLayoutNames);
    else if (t.param == "noise_source") same(t.names, s::kNoiseSourceNames);
    else if (t.param == "resize") same(t.names, s::kResizeNames);
    else if (t.param == "mask_encoding") same(t.names, s::kMaskEncodingNames);
    else if (t.param == "step_cache") same(t.names, s::kStepCacheNames);
    else if (t.param == "lora_placement") same(t.names, s::kLoraPlacementNames);
    else ADD_FAILURE() << "unexpected parameter";
  }
}

// The report of a done event: a NULL from mi_result_report_json() becomes JSON null instead of a crash.
TEST(ServerJson, OneLine) {
  EXPECT_EQ(mi::server::one_line(nullptr), "null");
  EXPECT_EQ(mi::server::one_line("{\n \"a\": \"x\\ny\",\r\n \"b\": 1\n}\n"), "{ \"a\": \"x\\ny\", \"b\": 1}");
}

}  // namespace
