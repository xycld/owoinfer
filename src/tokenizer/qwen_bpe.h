// Qwen2 byte-level BPE tokenizer (HF `tokenizers` semantics) and the MOSS-Image chat template.
//
// encode() reproduces Tokenizer.encode(text, add_special_tokens=False) for the tokenizer.json
// shipped with MOSS-Image: added tokens are split out of the raw text first, the remaining
// segments are NFC-normalized, split by the Qwen2 pre-tokenizer regex, byte-level mapped and BPE
// merged. load() rejects tokenizer.json files whose pipeline differs from that.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "util/json.h"

namespace mi::tok {

// Bounds on a user-supplied tokenizer.json (the Qwen one is 20 MB with 1.24M JSON values).
inline constexpr std::uintmax_t kMaxTokenizerBytes = std::uintmax_t{64} << 20;
inline constexpr std::size_t kMaxTokenizerValues = 4'000'000;

inline constexpr std::size_t kMossMaxSequenceLength = 4096;

struct AddedToken {
  std::string content;
  std::int32_t id = 0;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = false;
  bool special = false;
};

class Tokenizer {
 public:
  // Throws std::runtime_error on unreadable or unsupported files.
  static Tokenizer load(const std::filesystem::path& tokenizer_json);
  static Tokenizer from_json(const json::Value& root);

  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  ~Tokenizer();

  // Throws std::invalid_argument if `text` is not valid UTF-8 (HF cannot receive such input at
  // all: a Python str always holds valid Unicode scalar values). Thread-safe.
  std::vector<std::int32_t> encode(std::string_view text) const;

  // One past the largest token id, over the BPE vocabulary and the added tokens: every id encode()
  // returns is below this, so an embedding table needs at least this many rows.
  std::size_t vocab_size() const;
  const std::vector<AddedToken>& added_tokens() const;

 private:
  struct Impl;
  explicit Tokenizer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// The Qwen2 Split pre-tokenizer alone (no added tokens, no NFC): the regex matches tiling `text`.
// Throws std::invalid_argument for invalid UTF-8.
std::vector<std::string_view> qwen2_pretokenize(std::string_view text);

// Python str.strip(): removes leading/trailing characters for which str.isspace() is true.
// `s` must be valid UTF-8.
std::string_view python_strip(std::string_view s);

// The official template, apply_chat_template([{role: user, content: prompt}],
// add_generation_prompt=True, enable_thinking=True):
//   "<|im_start|>user\n" + prompt.strip() + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
// Throws std::invalid_argument for invalid UTF-8, and for prompts the template itself rejects
// (a stripped prompt that starts with <tool_response> and ends with </tool_response>).
std::string format_moss_prompt(std::string_view prompt);

// The pipeline's tokenizer(format_moss_prompt(prompt), truncation=True, max_length=max_length):
// no special tokens are added, and truncation keeps the first max_length ids.
std::vector<std::int32_t> encode_moss_prompt(const Tokenizer& tokenizer, std::string_view prompt,
                                             std::size_t max_length = kMossMaxSequenceLength);

}  // namespace mi::tok
