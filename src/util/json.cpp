#include "util/json.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <string_view>

namespace mi::json {

namespace {

[[noreturn]] void fail(const char* what, std::size_t pos) {
  throw ParseError(std::string("json: ") + what + " at byte " + std::to_string(pos));
}

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// For a valid JSON number text that from_chars reported out of range: true when the magnitude is
// below the smallest subnormal (decimal exponent of the first nonzero digit is negative), false
// when it is too large.
bool underflows(std::string_view text) {
  std::size_t i = text.front() == '-' ? 1 : 0;
  std::int64_t int_digits = 0, first_nonzero = -1, index = 0;
  bool after_point = false;
  for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
    if (text[i] == '.') {
      after_point = true;
      continue;
    }
    if (!after_point) ++int_digits;
    if (first_nonzero < 0 && text[i] != '0') first_nonzero = index;
    ++index;
  }
  std::int64_t exp = 0;
  if (i < text.size()) {
    ++i;
    const bool neg = text[i] == '-';
    if (text[i] == '+' || text[i] == '-') ++i;
    for (; i < text.size(); ++i) exp = std::min<std::int64_t>(exp * 10 + (text[i] - '0'), std::int64_t{1} << 40);  // saturate
    if (neg) exp = -exp;
  }
  return int_digits - first_nonzero - 1 + exp < 0;
}

class Parser {
 public:
  Parser(std::string_view s, const ParseLimits& limits) : s_(s), limits_(limits) {}

  Value document() {
    Value v = value(0);
    skip_ws();
    if (pos_ != s_.size()) fail("trailing characters", pos_);
    return v;
  }

 private:
  // Each level costs a few hundred bytes of stack, more in MSVC debug builds.
  static constexpr int kMaxDepth = 128;

  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }

  char peek() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of input", pos_);
    return s_[pos_];
  }

  void expect_literal(std::string_view lit) {
    if (s_.substr(pos_, lit.size()) != lit) fail("invalid literal", pos_);
    pos_ += lit.size();
  }

  Value value(int depth) {
    if (depth >= kMaxDepth) fail("nesting too deep", pos_);
    if (++values_ > limits_.max_values) fail("too many values", pos_);
    switch (peek()) {
      case '{': return object(depth);
      case '[': return array(depth);
      case '"': return Value(string());
      case 't': expect_literal("true"); return Value(true);
      case 'f': expect_literal("false"); return Value(false);
      case 'n': expect_literal("null"); return Value();
      default: return number();
    }
  }

  Value object(int depth) {
    ++pos_;  // '{'
    Object obj;
    if (peek() == '}') {
      ++pos_;
      return Value(std::move(obj));
    }
    for (;;) {
      if (peek() != '"') fail("expected object key", pos_);
      std::string key = string();
      if (peek() != ':') fail("expected ':'", pos_);
      ++pos_;
      obj.emplace_back(std::move(key), value(depth + 1));
      char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'", pos_ - 1);
    }
    return Value(std::move(obj));
  }

  Value array(int depth) {
    ++pos_;  // '['
    Array arr;
    if (peek() == ']') {
      ++pos_;
      return Value(std::move(arr));
    }
    for (;;) {
      arr.push_back(value(depth + 1));
      char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'", pos_ - 1);
    }
    return Value(std::move(arr));
  }

  std::uint32_t hex4() {
    if (pos_ + 4 > s_.size()) fail("truncated \\u escape", pos_);
    std::uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<std::uint32_t>(c - 'A' + 10);
      else fail("invalid hex digit", pos_ - 1);
    }
    return v;
  }

  std::string string() {
    ++pos_;  // opening quote
    std::string out;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string", pos_);
      // Copy the run of plain characters in one go (tokenizer.json is ~20 MB).
      std::size_t start = pos_;
      while (pos_ < s_.size() && s_[pos_] != '"' && s_[pos_] != '\\') {
        if (static_cast<unsigned char>(s_[pos_]) < 0x20) fail("control character in string", pos_);
        ++pos_;
      }
      out.append(s_.substr(start, pos_ - start));
      if (pos_ >= s_.size()) fail("unterminated string", pos_);
      if (s_[pos_] == '"') {
        ++pos_;
        return out;
      }
      ++pos_;  // backslash
      if (pos_ >= s_.size()) fail("unterminated escape", pos_);
      char e = s_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          std::uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (s_.substr(pos_, 2) != "\\u") fail("unpaired high surrogate", pos_);
            pos_ += 2;
            std::uint32_t lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate", pos_ - 4);
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("unpaired low surrogate", pos_ - 4);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("invalid escape", pos_ - 1);
      }
    }
  }

  Value number() {
    std::size_t start = pos_;
    bool is_int = true;
    if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
    if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("invalid number", start);
    if (s_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    if (pos_ < s_.size() && s_[pos_] == '.') {
      is_int = false;
      ++pos_;
      if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("invalid fraction", pos_);
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      is_int = false;
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
      if (pos_ >= s_.size() || !(s_[pos_] >= '0' && s_[pos_] <= '9')) fail("invalid exponent", pos_);
      while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
    }
    std::string_view text = s_.substr(start, pos_ - start);
    std::int64_t i = 0;
    if (is_int) {
      auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), i);
      if (ec != std::errc() || ptr != text.data() + text.size()) is_int = false;  // out of int64 range
    }
    // from_chars is locale-independent (the C library's parsers honor LC_NUMERIC, so under a
    // comma-decimal locale "0.5" would stop at the '.') and needs no NUL-terminated copy.
    double d = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), d);
    if (ec == std::errc::result_out_of_range && underflows(text)) {
      d = text.front() == '-' ? -0.0 : 0.0;  // like Python's json: a magnitude below the subnormals is zero
      ec = std::errc();
      ptr = text.data() + text.size();
    }
    if (ec != std::errc() || ptr != text.data() + text.size()) fail("number out of double range", start);
    return Value(d, is_int, i);
  }

  std::string_view s_;
  ParseLimits limits_;
  std::size_t pos_ = 0;
  std::size_t values_ = 0;
};

[[noreturn]] void type_error(const char* want) {
  throw std::runtime_error(std::string("json: value is not ") + want);
}

}  // namespace

bool Value::as_bool() const {
  if (!is_bool()) type_error("a bool");
  return std::get<bool>(data_);
}

double Value::as_double() const {
  if (!is_number()) type_error("a number");
  return std::get<Number>(data_).d;
}

std::int64_t Value::as_int() const {
  if (!is_number() || !std::get<Number>(data_).is_int) type_error("an integer");
  return std::get<Number>(data_).i;
}

const std::string& Value::as_string() const {
  if (!is_string()) type_error("a string");
  return std::get<std::string>(data_);
}

const Array& Value::as_array() const {
  if (!is_array()) type_error("an array");
  return std::get<Array>(data_);
}

const Object& Value::as_object() const {
  if (!is_object()) type_error("an object");
  return std::get<Object>(data_);
}

const Value* Value::find(std::string_view key) const {
  if (!is_object()) return nullptr;
  for (const auto& [k, v] : std::get<Object>(data_)) {
    if (k == key) return &v;
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr) throw std::runtime_error("json: missing key '" + std::string(key) + "'");
  return *v;
}

Value parse(std::string_view text, const ParseLimits& limits) { return Parser(text, limits).document(); }

}  // namespace mi::json
