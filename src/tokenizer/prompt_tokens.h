// Prompt token count and truncation report for the MOSS-Image text encoder input.
//
// The official pipeline (MOSSImagePipeline.encode_prompt) formats the prompt with the chat template, then tokenizes
// with truncation=True, max_length=max_sequence_length: it keeps the first max_length ids of the formatted sequence
// and silently drops the rest, so the end of a long prompt and the template's closing tokens
// ("<|im_end|>\n<|im_start|>assistant\n<think>\n") are cut. This report lets a front end show the count and warn.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "tokenizer/qwen_bpe.h"

namespace mi::tok {

struct PromptTokenReport {
  std::size_t total_tokens = 0;     // formatted sequence (template included) before truncation
  std::size_t kept_tokens = 0;      // min(total_tokens, max_length): what the text encoder receives
  std::size_t template_tokens = 0;  // the template's own tokens (the formatted empty prompt)
  std::size_t max_length = 0;
  bool truncated = false;           // total_tokens > max_length
  // Tokens of the prompt text itself: total_tokens - template_tokens (the template's pieces are split off as special
  // tokens and newlines, which a stripped prompt does not merge with).
  std::size_t prompt_tokens() const { return total_tokens - template_tokens; }
  std::size_t dropped_tokens() const { return total_tokens - kept_tokens; }
};

// encode_moss_prompt() plus the report. Throws std::invalid_argument like encode_moss_prompt (invalid UTF-8, a
// prompt the template rejects) and for max_length == 0.
std::vector<std::int32_t> encode_moss_prompt_report(const Tokenizer& tokenizer, std::string_view prompt,
                                                    std::size_t max_length, PromptTokenReport& report);

// The report alone.
PromptTokenReport count_moss_prompt_tokens(const Tokenizer& tokenizer, std::string_view prompt,
                                           std::size_t max_length = kMossMaxSequenceLength);

}  // namespace mi::tok
