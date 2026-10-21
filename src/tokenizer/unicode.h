// Unicode support for the Qwen tokenizer: strict UTF-8, the character classes of the pre-tokenizer
// regex, and NFC. Tables come from tools/gen/gen_unicode_tables.py and reproduce the HF
// `tokenizers` library, whose regex engine and normalizer use *different* Unicode versions.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mi::tok {

// Bits returned by regex_class(): the \p{L}, \p{N} and \s classes of the Oniguruma regex engine.
inline constexpr std::uint8_t kClassL = 1;
inline constexpr std::uint8_t kClassN = 2;
inline constexpr std::uint8_t kClassS = 4;

// Throws std::invalid_argument naming the byte offset of the first malformed sequence (overlong
// forms, surrogates, code points above U+10FFFF and truncated sequences are all malformed).
void validate_utf8(std::string_view s);

// Decodes the code point starting at s[i] and advances i. `s` must be valid UTF-8.
char32_t next_code_point(std::string_view s, std::size_t& i);
void append_utf8(std::string& out, char32_t cp);

std::uint8_t regex_class(char32_t cp);
// The word-character test of added tokens with single_word=true.
bool is_word_char(char32_t cp);
// Python str.isspace().
bool is_python_space(char32_t cp);

// NFC exactly as the `tokenizers` NFC normalizer computes it. `s` must be valid UTF-8.
std::string nfc(std::string_view s);

}  // namespace mi::tok
