#include <gtest/gtest.h>

#include <bit>
#include <charconv>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include "util/json.h"

using mi::json::parse;
using mi::json::ParseError;

TEST(Json, ScalarsAndNesting) {
  auto v = parse(R"({"a": [1, -2.5e2, true, false, null], "b": {"c": "x"}, "big": 9223372036854775807})");
  const auto& a = v.at("a").as_array();
  ASSERT_EQ(a.size(), 5u);
  EXPECT_EQ(a[0].as_int(), 1);
  EXPECT_DOUBLE_EQ(a[1].as_double(), -250.0);
  EXPECT_THROW(a[1].as_int(), std::runtime_error);
  EXPECT_TRUE(a[2].as_bool());
  EXPECT_FALSE(a[3].as_bool());
  EXPECT_TRUE(a[4].is_null());
  EXPECT_EQ(v.at("b").at("c").as_string(), "x");
  EXPECT_EQ(v.at("big").as_int(), INT64_MAX);
  EXPECT_EQ(v.find("missing"), nullptr);
}

TEST(Json, ObjectKeyOrderPreserved) {
  auto v = parse(R"({"z": 1, "a": 2, "m": 3})");
  const auto& o = v.as_object();
  ASSERT_EQ(o.size(), 3u);
  EXPECT_EQ(o[0].first, "z");
  EXPECT_EQ(o[1].first, "a");
  EXPECT_EQ(o[2].first, "m");
}

TEST(Json, StringEscapesAndUnicode) {
  auto v = parse(R"(["a\"b\\c\/\n\t", "\u00e9\u4e2d", "\ud83d\ude00", "Ġhello"])");
  const auto& a = v.as_array();
  EXPECT_EQ(a[0].as_string(), "a\"b\\c/\n\t");
  EXPECT_EQ(a[1].as_string(), "\xC3\xA9\xE4\xB8\xAD");
  EXPECT_EQ(a[2].as_string(), "\xF0\x9F\x98\x80");
  EXPECT_EQ(a[3].as_string(), "\xC4\xA0hello");
}

TEST(Json, RejectsMalformed) {
  for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "\"\\ud800\"", "01", "1.", "tru", "[1] x",
                          "\"a\x01\"", "{\"a\":1,}"}) {
    EXPECT_THROW(parse(bad), ParseError) << bad;
  }
}

TEST(Json, NestingLimitIs128) {
  auto nested = [](int depth) { return std::string(static_cast<std::size_t>(depth), '[') + std::string(static_cast<std::size_t>(depth), ']'); };
  EXPECT_NO_THROW(parse(nested(128)));
  EXPECT_THROW(parse(nested(129)), ParseError);
  EXPECT_THROW(parse(std::string(100000, '[')), ParseError);  // fails fast, no deep recursion
}

TEST(Json, ValueLimit) {
  EXPECT_NO_THROW(parse("[1,2,3]", {.max_values = 4}));  // the array plus three numbers
  EXPECT_THROW(parse("[1,2,3]", {.max_values = 3}), ParseError);
  EXPECT_THROW(parse(R"({"a":{"b":[true,null]}})", {.max_values = 4}), ParseError);
}

TEST(Json, RejectsNumbersOutsideDoubleRange) {
  EXPECT_THROW(parse("1e400"), ParseError);
  EXPECT_THROW(parse("-1e400"), ParseError);
}

// Python's json maps magnitude underflow to a signed zero; only overflow is an error.
TEST(Json, UnderflowParsesAsSignedZero) {
  for (const char* text : {"1e-400", "0.000001e-320", "123456789e-999", "1e-99999999999999999999"}) {
    const double d = parse(text).as_double();
    EXPECT_EQ(d, 0.0) << text;
    EXPECT_FALSE(std::signbit(d)) << text;
    const double n = parse(std::string("-") + text).as_double();
    EXPECT_EQ(n, 0.0) << text;
    EXPECT_TRUE(std::signbit(n)) << text;
  }
  EXPECT_FALSE(parse("1e-400").is_int());
  // Negative exponent but still too large: 1 followed by 500 zeros, times 1e-100.
  EXPECT_THROW(parse("1" + std::string(500, '0') + "e-100"), ParseError);
  EXPECT_THROW(parse("1e400"), ParseError);
  EXPECT_THROW(parse("-1e400"), ParseError);
  EXPECT_EQ(parse("5e-324").as_double(), 5e-324);  // smallest subnormal is still exact
}

namespace {

// Number texts whose parse must equal std::from_chars exactly (round-trip-hard values, subnormals,
// extreme exponents).
constexpr const char* kNumberTexts[] = {
    "0.5", "0.1", "-0.1", "1e-7", "1e-06", "2.5", "-2.5e+300", "5e-324", "2.2250738585072014e-308",
    "1.7976931348623157e308", "0.30000000000000004", "123456789.123456789", "9007199254740993", "-0.0",
    "1E+2", "18446744073709551616", "3.141592653589793238462643383279",
};

double from_chars_double(std::string_view s) {
  double d = 0;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), d);
  EXPECT_TRUE(ec == std::errc() && ptr == s.data() + s.size()) << s;
  return d;
}

void expect_exact_numbers() {
  for (const char* text : kNumberTexts) {
    const double got = parse(text).as_double();
    EXPECT_EQ(std::bit_cast<std::uint64_t>(got), std::bit_cast<std::uint64_t>(from_chars_double(text))) << text;
  }
  EXPECT_EQ(parse("0.5").as_double(), 0.5);
  EXPECT_EQ(parse("[1.25]").as_array()[0].as_double(), 1.25);
}

}  // namespace

// The C library's number parsers honor LC_NUMERIC: under a comma-decimal locale they stop at the
// '.', so "0.5" would parse as 0. The parser must not depend on the process locale (a desktop app's
// UI toolkit commonly calls setlocale(LC_ALL, "")).
TEST(Json, NumbersIgnoreProcessLocale) {
  expect_exact_numbers();

  // Locale-independence cannot be shown on a machine without a comma-decimal locale, so the source is
  // also checked to stay off the locale-dependent conversions.
  std::ifstream src(std::string(MOSSINFER_SOURCE_DIR) + "/src/util/json.cpp");
  ASSERT_TRUE(src.good());
  const std::string code((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
  for (const char* banned : {"strtod", "strtof", "strtold", "atof", "stod(", "stof(", "sscanf", "istringstream"}) {
    EXPECT_EQ(code.find(banned), std::string::npos) << "json.cpp uses locale-dependent " << banned;
  }

  const std::string saved = std::setlocale(LC_ALL, nullptr);
  const char* used = nullptr;
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "German_Germany.1252"}) {
    if (std::setlocale(LC_ALL, name) != nullptr) {
      used = name;
      break;
    }
  }
  if (used != nullptr) {
    EXPECT_STREQ(std::localeconv()->decimal_point, ",") << used;
    expect_exact_numbers();
    std::setlocale(LC_ALL, saved.c_str());
  }
  RecordProperty("comma_locale", used != nullptr ? used : "none installed");
  std::printf("comma-decimal locale for this test: %s\n", used != nullptr ? used : "none installed (source check only)");
}
