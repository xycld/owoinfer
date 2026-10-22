// Lane D (P7a): prompt token count and truncation report against HF tokenizers run as MOSSImagePipeline.encode_prompt
// does (tests/data/tokenizer_truncation.jsonl): chat template, then truncation=True, max_length.
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tokenizer/prompt_tokens.h"
#include "util/json.h"

namespace {

namespace fs = std::filesystem;

fs::path tokenizer_json() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  const fs::path weights = env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
  return weights / "base/tokenizer/tokenizer.json";
}

std::vector<mi::json::Value> read_jsonl(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<mi::json::Value> rows;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) rows.push_back(mi::json::parse(line));
  }
  return rows;
}

TEST(TokenizerTruncation, CountsAndTruncationMatchReference) {
  if (!fs::is_regular_file(tokenizer_json())) GTEST_SKIP() << tokenizer_json() << " absent";
  const mi::tok::Tokenizer tok = mi::tok::Tokenizer::load(tokenizer_json());
  const auto rows = read_jsonl(fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/tokenizer_truncation.jsonl");
  ASSERT_GT(rows.size(), 1u);
  const auto& prompts = rows[0].at("prompts").as_array();
  const auto template_tokens = static_cast<std::size_t>(rows[0].at("template").as_int());
  int cases = 0, truncated = 0;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const auto& r = rows[i];
    ASSERT_TRUE(r.at("prefix_ok").as_bool());  // the reference keeps the first max_length ids
    const std::string& prompt = prompts[static_cast<std::size_t>(r.at("prompt").as_int())].as_string();
    const auto max_length = static_cast<std::size_t>(r.at("max_length").as_int());
    mi::tok::PromptTokenReport rep;
    const std::vector<std::int32_t> ids = mi::tok::encode_moss_prompt_report(tok, prompt, max_length, rep);
    const std::string what = "prompt of " + std::to_string(prompt.size()) + " bytes, max_length " + std::to_string(max_length);
    EXPECT_EQ(ids, mi::tok::encode_moss_prompt(tok, prompt, max_length)) << what;
    EXPECT_EQ(rep.total_tokens, static_cast<std::size_t>(r.at("total").as_int())) << what;
    EXPECT_EQ(rep.kept_tokens, static_cast<std::size_t>(r.at("kept").as_int())) << what;
    EXPECT_EQ(rep.kept_tokens, ids.size()) << what;
    EXPECT_EQ(rep.template_tokens, template_tokens) << what;
    EXPECT_EQ(rep.truncated, rep.total_tokens > max_length) << what;
    EXPECT_EQ(rep.max_length, max_length);
    EXPECT_EQ(rep.dropped_tokens(), rep.total_tokens - rep.kept_tokens);
    std::vector<std::int32_t> last;
    for (const auto& v : r.at("last_ids").as_array()) last.push_back(static_cast<std::int32_t>(v.as_int()));
    ASSERT_GE(ids.size(), last.size());
    EXPECT_EQ(std::vector<std::int32_t>(ids.end() - static_cast<std::ptrdiff_t>(last.size()), ids.end()), last) << what;
    const mi::tok::PromptTokenReport again = mi::tok::count_moss_prompt_tokens(tok, prompt, max_length);
    EXPECT_EQ(again.total_tokens, rep.total_tokens);
    ++cases;
    truncated += rep.truncated;
  }
  EXPECT_EQ(cases, 42);
  EXPECT_GT(truncated, 0);
}

TEST(TokenizerTruncation, LargeChinesePromptAndErrors) {
  if (!fs::is_regular_file(tokenizer_json())) GTEST_SKIP() << tokenizer_json() << " absent";
  const mi::tok::Tokenizer tok = mi::tok::Tokenizer::load(tokenizer_json());
  const auto rows = read_jsonl(fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/tokenizer_truncation.jsonl");
  const std::string& chinese = rows[0].at("prompts").as_array()[4].as_string();  // 9000 characters
  const mi::tok::PromptTokenReport r = mi::tok::count_moss_prompt_tokens(tok, chinese);
  EXPECT_EQ(r.max_length, mi::tok::kMossMaxSequenceLength);
  EXPECT_TRUE(r.truncated);
  EXPECT_EQ(r.kept_tokens, mi::tok::kMossMaxSequenceLength);
  EXPECT_GT(r.prompt_tokens(), 5000u);
  EXPECT_THROW(mi::tok::count_moss_prompt_tokens(tok, "a cat", 0), std::invalid_argument);
  EXPECT_THROW(mi::tok::count_moss_prompt_tokens(tok, std::string("\xff\xfe", 2)), std::invalid_argument);
  const mi::tok::PromptTokenReport empty = mi::tok::count_moss_prompt_tokens(tok, "");
  EXPECT_EQ(empty.prompt_tokens(), 0u);
  EXPECT_FALSE(empty.truncated);
}

}  // namespace
