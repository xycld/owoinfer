#include "tokenizer/prompt_tokens.h"

#include <stdexcept>

namespace mi::tok {

std::vector<std::int32_t> encode_moss_prompt_report(const Tokenizer& tokenizer, std::string_view prompt,
                                                    std::size_t max_length, PromptTokenReport& report) {
  if (max_length == 0) throw std::invalid_argument("tokenizer: max_length must be >= 1");
  std::vector<std::int32_t> ids = tokenizer.encode(format_moss_prompt(prompt));
  PromptTokenReport r;
  r.total_tokens = ids.size();
  r.max_length = max_length;
  r.template_tokens = tokenizer.encode(format_moss_prompt("")).size();
  r.truncated = ids.size() > max_length;
  if (r.truncated) ids.resize(max_length);
  r.kept_tokens = ids.size();
  report = r;
  return ids;
}

PromptTokenReport count_moss_prompt_tokens(const Tokenizer& tokenizer, std::string_view prompt, std::size_t max_length) {
  PromptTokenReport r;
  encode_moss_prompt_report(tokenizer, prompt, max_length, r);
  return r;
}

}  // namespace mi::tok
