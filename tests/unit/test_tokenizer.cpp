#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.h"
#include "tokenizer/qwen_bpe.h"
#include "tokenizer/unicode.h"
#include "util/json.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
using mi::tok::Tokenizer;
using Ids = std::vector<std::int32_t>;

const fs::path kSource = fs::path(MOSSINFER_SOURCE_DIR);

fs::path tokenizer_json() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  fs::path weights = env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
  return weights / "base/tokenizer/tokenizer.json";
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<mi::json::Value> read_jsonl(const fs::path& path) {
  std::vector<mi::json::Value> rows;
  std::istringstream in(read_file(path));
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) rows.push_back(mi::json::parse(line));
  }
  return rows;
}

Ids ids_of(const mi::json::Value& array) {
  Ids out;
  for (const mi::json::Value& v : array.as_array()) out.push_back(static_cast<std::int32_t>(v.as_int()));
  return out;
}

Ids npy_ids(const fs::path& path) {
  const std::vector<std::int64_t> v = mi::npy::load(path).to_vector<std::int64_t>();
  return Ids(v.begin(), v.end());
}

std::string escape(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    if (c < 0x20 || c == 0x7F) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\x%02X", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

std::string ids_str(const Ids& ids) {
  std::string s;
  for (std::int32_t id : ids) s += std::to_string(id) + ' ';
  return s;
}

class TokenizerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::exists(tokenizer_json())) return;
    auto t0 = std::chrono::steady_clock::now();
    tok_ = std::make_unique<Tokenizer>(Tokenizer::load(tokenizer_json()));
    load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  static void TearDownTestSuite() { tok_.reset(); }
  void SetUp() override {
    if (!tok_) GTEST_SKIP() << "weights not found: " << tokenizer_json();
  }

  static std::unique_ptr<Tokenizer> tok_;
  static double load_ms_;
};
std::unique_ptr<Tokenizer> TokenizerTest::tok_;
double TokenizerTest::load_ms_ = 0;

TEST_F(TokenizerTest, LoadsQwenVocabulary) {
  EXPECT_EQ(tok_->vocab_size(), 248077u);
  ASSERT_EQ(tok_->added_tokens().size(), 33u);
  EXPECT_EQ(tok_->added_tokens()[1].content, "<|im_start|>");
  EXPECT_EQ(tok_->added_tokens()[1].id, 248045);
  std::printf("tokenizer.json load: %.0f ms\n", load_ms_);
  RecordProperty("load_ms", std::to_string(load_ms_));
  if constexpr (mi::testing::kAssertTimings) {
    EXPECT_LT(load_ms_, 1500.0);
  }
}

TEST_F(TokenizerTest, GoldenTokenIds) {
  const fs::path golden = kSource / "tests/golden";
  mi::json::Value cases = mi::json::parse(read_file(golden / "cases.json"));
  int checked = 0;
  for (const auto& entry : fs::directory_iterator(golden)) {
    std::string name = entry.path().filename().string();
    const std::string suffix = ".formatted.txt";
    if (!name.ends_with(suffix)) continue;
    std::string stem = name.substr(0, name.size() - suffix.size());  // <case>.<cond|uncond>
    SCOPED_TRACE(stem);
    Ids want = npy_ids(golden / (stem + ".token_ids.npy"));
    std::string formatted = read_file(entry.path());
    Ids got = tok_->encode(formatted);
    if (got.size() > mi::tok::kMossMaxSequenceLength) got.resize(mi::tok::kMossMaxSequenceLength);
    EXPECT_EQ(got, want);

    std::string id = stem.substr(0, stem.rfind('.'));
    bool cond = stem.ends_with(".cond");
    for (const mi::json::Value& c : cases.as_array()) {
      if (c.at("id").as_string() != id) continue;
      const std::string& prompt = c.at(cond ? "prompt" : "negative_prompt").as_string();
      auto max_length = static_cast<std::size_t>(c.at("max_sequence_length").as_int());
      EXPECT_EQ(mi::tok::format_moss_prompt(prompt), formatted);
      EXPECT_EQ(mi::tok::encode_moss_prompt(*tok_, prompt, max_length), want);
    }
    ++checked;
  }
  EXPECT_EQ(checked, 14);
}

TEST_F(TokenizerTest, OfficialPromptsAndTemplateEdgeCases) {
  int assets = 0;
  for (const mi::json::Value& row : read_jsonl(kSource / "tests/data/tokenizer_prompts.jsonl")) {
    const std::string& name = row.at("name").as_string();
    const std::string& prompt = row.at("prompt").as_string();
    SCOPED_TRACE(name);
    if (row.find("error") != nullptr) {
      EXPECT_THROW(mi::tok::format_moss_prompt(prompt), std::invalid_argument);
      continue;
    }
    auto max_length = static_cast<std::size_t>(row.at("max_length").as_int());
    EXPECT_EQ(mi::tok::format_moss_prompt(prompt), row.at("formatted").as_string());
    EXPECT_EQ(mi::tok::encode_moss_prompt(*tok_, prompt, max_length), ids_of(row.at("ids")));
    assets += name.starts_with("assets/");
  }
  EXPECT_EQ(assets, 40);  // 20 prompts + 20 negative prompts
}

TEST_F(TokenizerTest, FuzzMatchesHuggingFace) {
  std::vector<mi::json::Value> rows = read_jsonl(kSource / "tests/data/tokenizer_fuzz.jsonl");
  ASSERT_EQ(rows.size(), 10000u);
  int matches = 0, reported = 0;
  for (const mi::json::Value& row : rows) {
    const std::string& text = row.at("t").as_string();
    Ids want = ids_of(row.at("ids"));
    Ids got = tok_->encode(text);
    if (got == want) {
      ++matches;
    } else if (reported++ < 10) {
      ADD_FAILURE() << "text: \"" << escape(text) << "\"\n  want: " << ids_str(want) << "\n  got:  " << ids_str(got);
    }
  }
  EXPECT_EQ(matches, 10000);
}

TEST_F(TokenizerTest, AddedTokenFlagsMatchTokenizersLibrary) {
  std::vector<mi::json::Value> rows = read_jsonl(kSource / "tests/data/tokenizer_flags.jsonl");
  ASSERT_GT(rows.size(), 1u);
  // Line 1 holds the variant's added_tokens; splice them into the real tokenizer.json.
  mi::json::Value base = mi::json::parse(read_file(tokenizer_json()));
  mi::json::Object variant;
  for (const auto& [key, value] : base.as_object()) {
    variant.emplace_back(key, key == "added_tokens" ? rows[0].at("added_tokens") : value);
  }
  Tokenizer tok = Tokenizer::from_json(mi::json::Value(std::move(variant)));
  int matches = 0;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    const std::string& text = rows[i].at("t").as_string();
    Ids want = ids_of(rows[i].at("ids"));
    Ids got = tok.encode(text);
    EXPECT_EQ(got, want) << "text: \"" << escape(text) << '"';
    matches += got == want;
  }
  EXPECT_EQ(matches, static_cast<int>(rows.size() - 1));
}

TEST_F(TokenizerTest, RejectsInvalidUtf8) {
  const std::string bad[] = {
      "\x80",              // stray continuation byte
      "abc\xC3",           // truncated 2-byte sequence
      "\xC0\xAF",          // overlong '/'
      "\xE0\x80\xAF",      // overlong 3-byte
      "\xED\xA0\x80",      // UTF-16 surrogate U+D800
      "\xF4\x90\x80\x80",  // above U+10FFFF
      "\xF8\x88\x80\x80\x80",
      "ok \xE4\xB8 broken",  // truncated CJK
  };
  for (const std::string& s : bad) {
    EXPECT_THROW(tok_->encode(s), std::invalid_argument) << escape(s);
    EXPECT_THROW(mi::tok::format_moss_prompt(s), std::invalid_argument) << escape(s);
  }
  // Boundary scalar values are valid.
  EXPECT_NO_THROW(tok_->encode("\xED\x9F\xBF\xEE\x80\x80\xEF\xBF\xBF\xF4\x8F\xBF\xBF"));
}

TEST_F(TokenizerTest, LongestAssetPromptEncodesFast) {
  std::string longest;
  for (const mi::json::Value& row : read_jsonl(kSource / "tests/data/tokenizer_prompts.jsonl")) {
    const std::string& p = row.at("prompt").as_string();
    if (p.size() > longest.size()) longest = p;
  }
  Ids warm = mi::tok::encode_moss_prompt(*tok_, longest);
  double best_ms = 1e9;
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    Ids ids = mi::tok::encode_moss_prompt(*tok_, longest);
    best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ASSERT_EQ(ids, warm);
  }
  std::printf("encode %zu bytes -> %zu ids: %.2f ms\n", longest.size(), warm.size(), best_ms);
  RecordProperty("encode_ms", std::to_string(best_ms));
  if constexpr (mi::testing::kAssertTimings) {
    EXPECT_LT(best_ms, 20.0);
  }
}

TEST(TokenizerPretokenize, MatchesQwen2SplitRegex) {
  int matches = 0, total = 0;
  for (const mi::json::Value& row : read_jsonl(kSource / "tests/data/tokenizer_pretok.jsonl")) {
    const std::string& text = row.at("t").as_string();
    std::vector<std::int64_t> want;
    for (const mi::json::Value& n : row.at("n").as_array()) want.push_back(n.as_int());
    std::vector<std::int64_t> got;
    for (std::string_view piece : mi::tok::qwen2_pretokenize(text)) got.push_back(static_cast<std::int64_t>(piece.size()));
    EXPECT_EQ(got, want) << "text: \"" << escape(text) << '"';
    matches += got == want;
    ++total;
  }
  EXPECT_EQ(total, 4000);
  EXPECT_EQ(matches, total);
}

TEST(TokenizerTemplate, PythonStripUsesStrIsspace) {
  using mi::tok::python_strip;
  EXPECT_EQ(python_strip("  a b \n"), "a b");
  EXPECT_EQ(python_strip("\x1c\x1d" "a" "\x1e\x1f"), "a");                // str.isspace, not \s
  EXPECT_EQ(python_strip("\xE3\x80\x80" "a" "\xC2\xA0"), "a");             // U+3000, U+00A0
  EXPECT_EQ(python_strip("\xE2\x80\x8B" "a"), "\xE2\x80\x8B" "a");         // U+200B is not a space
  EXPECT_EQ(python_strip("\xE1\xA0\x8E" "a"), "\xE1\xA0\x8E" "a");         // U+180E is not a space
  EXPECT_EQ(python_strip(" \t\r\n\v\f"), "");
  EXPECT_EQ(mi::tok::format_moss_prompt(""), "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n");
}

// Smallest tokenizer.json from_json accepts: Qwen2 pipeline, the 256 byte-level symbols as vocab
// ids 0-255, no merges, and the given added tokens.
mi::json::Value synthetic_tokenizer(mi::json::Array added_tokens) {
  using mi::json::Array;
  using mi::json::Object;
  using mi::json::Value;
  auto str = [](std::string v) { return Value(std::move(v)); };
  auto num = [](std::int64_t v) { return Value(static_cast<double>(v), true, v); };
  Object vocab;
  std::int64_t id = 0;
  for (unsigned b = 0, next = 256; b < 256; ++b) {
    bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    std::string symbol;
    mi::tok::append_utf8(symbol, printable ? static_cast<char32_t>(b) : static_cast<char32_t>(next++));
    vocab.emplace_back(symbol, num(id++));
  }
  Object split{{"type", str("Split")},
               {"pattern", Value(Object{{"Regex", str(R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)")}})},
               {"behavior", str("Isolated")},
               {"invert", Value(false)}};
  Object byte_level{{"type", str("ByteLevel")}, {"add_prefix_space", Value(false)}, {"use_regex", Value(false)}};
  Object root{{"added_tokens", Value(std::move(added_tokens))},
              {"normalizer", Value(Object{{"type", str("NFC")}})},
              {"pre_tokenizer", Value(Object{{"type", str("Sequence")}, {"pretokenizers", Value(Array{Value(std::move(split)), Value(std::move(byte_level))})}})},
              {"post_processor", Value()},
              {"model", Value(Object{{"type", str("BPE")}, {"vocab", Value(std::move(vocab))}, {"merges", Value(Array{})}})}};
  return Value(std::move(root));
}

mi::json::Value added_token(const std::string& content, std::int64_t id) {
  return mi::json::Value(mi::json::Object{{"content", mi::json::Value(content)}, {"id", mi::json::Value(static_cast<double>(id), true, id)}, {"special", mi::json::Value(true)}});
}

// tokenizer.json comes from a downloaded bundle: size is checked before reading, value count while parsing.
TEST(TokenizerLoad, RejectsOversizedAndOverValuedFiles) {
  mi::testing::TempDir dir("tok_caps");
  const fs::path path = dir.path() / "tokenizer.json";
  {
    std::ofstream(path, std::ios::binary) << "{}";
  }
  fs::resize_file(path, mi::tok::kMaxTokenizerBytes + 1);  // sparse; never read
  try {
    Tokenizer::load(path);
    ADD_FAILURE() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("over the limit"), std::string::npos) << e.what();
  }

  std::string many = "[0";
  many.reserve(2 * mi::tok::kMaxTokenizerValues + 8);
  for (std::size_t i = 1; i < mi::tok::kMaxTokenizerValues; ++i) many += ",0";
  many += "]";
  {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << many;
  }
  try {
    Tokenizer::load(path);
    ADD_FAILURE() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("too many values"), std::string::npos) << e.what();
  }
}

TEST(TokenizerAddedTokens, IdsMustFitInt32AndBeNonNegative) {
  Tokenizer ok = Tokenizer::from_json(synthetic_tokenizer({added_token("<a>", 300), added_token("<b>", 2147483647)}));
  EXPECT_EQ(ok.vocab_size(), 2147483648u);  // one past the largest id: an embedding table needs this many rows
  EXPECT_EQ(ok.encode("x<a>y<b>"), (Ids{'x', 300, 'y', 2147483647}));  // byte b has id b here
  EXPECT_EQ(Tokenizer::from_json(synthetic_tokenizer({})).vocab_size(), 256u);

  for (std::int64_t bad : {std::int64_t{-1}, std::int64_t{-2147483648LL}, std::int64_t{2147483648LL}, std::int64_t{4294967296LL + 300}}) {
    EXPECT_THROW(Tokenizer::from_json(synthetic_tokenizer({added_token("<a>", bad)})), std::runtime_error) << bad;
  }
}

TEST(TokenizerNfc, CanonicalOrderingOfCombiningMarks) {
  using mi::tok::nfc;
  // U+0301 (ccc 230) and U+0316 (ccc 220) reorder to 0316 0301; a + 0301 then composes to U+00E1.
  EXPECT_EQ(nfc("a\u0301\u0316"), "\u00E1\u0316");
  EXPECT_EQ(nfc("a\u0316\u0301"), "\u00E1\u0316");
  // Equal classes keep their order (stable), starters are never moved.
  EXPECT_EQ(nfc("q\u0301\u0308q\u0308\u0301"), "q\u0301\u0308q\u0308\u0301");
  EXPECT_EQ(nfc("\u0316\u0301a\u0301"), "\u0316\u0301\u00E1");  // leading marks have no starter
}

TEST(TokenizerNfc, LongCombiningRunIsNotQuadratic) {
  // 200k alternating marks of two classes: an insertion sort does ~5e9 swaps here.
  constexpr int kPairs = 100000;
  std::string text = "a";
  for (int i = 0; i < kPairs; ++i) text += "\u0301\u0316";
  const auto t0 = std::chrono::steady_clock::now();
  const std::string got = mi::tok::nfc(text);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::string want = "\u00E1";  // a + the first U+0301; the other U+0301s are blocked by equal class
  for (int i = 0; i < kPairs; ++i) want += "\u0316";
  for (int i = 1; i < kPairs; ++i) want += "\u0301";
  EXPECT_EQ(got, want);
  RecordProperty("nfc_200k_marks_ms", std::to_string(ms));
  if constexpr (mi::testing::kAssertTimings) {
    EXPECT_LT(ms, 500.0);
  }
}

TEST(TokenizerTemplate, RejectsUnsupportedPipelines) {
  // load() validates the pipeline before touching the vocabulary.
  const std::string other_regex = R"({"normalizer": {"type": "NFC"}, "pre_tokenizer": {"type": "Sequence",
      "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}]}, "model": {"type": "BPE"}})";
  EXPECT_THROW(Tokenizer::from_json(mi::json::parse(other_regex)), std::runtime_error);
  const std::string no_normalizer = R"({"normalizer": null, "model": {"type": "BPE"}})";
  EXPECT_THROW(Tokenizer::from_json(mi::json::parse(no_normalizer)), std::runtime_error);
}

}  // namespace
