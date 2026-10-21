// tools/server/json_out.h: moss-server's own JSON output helpers (it cannot link the pipeline's renderer) must write
// exactly what pipeline/report_json.h writes. Strings: every 1- and 2-byte input and the 3-byte inputs around each
// UTF-8 lead byte. Numbers: under a comma-decimal C++ locale, and a comma-decimal C locale where the system has one.
#include <gtest/gtest.h>

#include <clocale>
#include <cstdio>
#include <iterator>
#include <limits>
#include <locale>
#include <string>

#include "../../tools/server/json_out.h"
#include "pipeline/report_json.h"

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

// The report of a done event: a NULL from mi_result_report_json() becomes JSON null instead of a crash.
TEST(ServerJson, OneLine) {
  EXPECT_EQ(mi::server::one_line(nullptr), "null");
  EXPECT_EQ(mi::server::one_line("{\n \"a\": \"x\\ny\",\r\n \"b\": 1\n}\n"), "{ \"a\": \"x\\ny\", \"b\": 1}");
}

}  // namespace
