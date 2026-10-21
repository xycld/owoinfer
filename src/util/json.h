// Minimal JSON DOM parser (RFC 8259) for safetensors headers and tokenizer.json.
// Objects preserve key order; lookups are linear (callers index large objects themselves).
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mi::json {

class Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Value {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Value() = default;
  explicit Value(bool b) : data_(b) {}
  explicit Value(double d, bool is_int, std::int64_t i) : data_(Number{d, is_int, i}) {}
  explicit Value(std::string s) : data_(std::move(s)) {}
  explicit Value(Array a) : data_(std::move(a)) {}
  explicit Value(Object o) : data_(std::move(o)) {}

  Type type() const { return static_cast<Type>(data_.index()); }
  bool is_null() const { return type() == Type::kNull; }
  bool is_bool() const { return type() == Type::kBool; }
  bool is_number() const { return type() == Type::kNumber; }
  // A number written as an integer literal that fits in int64 (what as_int() accepts).
  bool is_int() const { return is_number() && std::get<Number>(data_).is_int; }
  bool is_string() const { return type() == Type::kString; }
  bool is_array() const { return type() == Type::kArray; }
  bool is_object() const { return type() == Type::kObject; }

  bool as_bool() const;
  double as_double() const;
  // Throws if the number is not an exact integer literal.
  std::int64_t as_int() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Object member lookup; returns nullptr if absent (or if this is not an object).
  const Value* find(std::string_view key) const;
  // Object member lookup; throws if absent.
  const Value& at(std::string_view key) const;

 private:
  struct Number {
    double d;
    bool is_int;
    std::int64_t i;
  };
  std::variant<std::monostate, bool, Number, std::string, Array, Object> data_;
};

// Bounds on what one document may build. Nesting is always limited to 128 levels (the parser is
// recursive); max_values caps the DOM's size, which can reach ~50x the text size, for untrusted input.
struct ParseLimits {
  std::size_t max_values = std::numeric_limits<std::size_t>::max();
};

// Parses a complete JSON document. Throws ParseError with a byte offset on malformed input or when
// a limit is exceeded.
Value parse(std::string_view text, const ParseLimits& limits = {});

}  // namespace mi::json
