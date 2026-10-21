#include "tokenizer/qwen_bpe.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "tokenizer/unicode.h"
#include "util/path.h"

namespace mi::tok {

namespace {

constexpr std::string_view kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Same bounds as the `tokenizers` BPE word cache (entries, and only words shorter than 256 bytes);
// they only affect speed and memory, never output.
constexpr std::size_t kCacheCapacity = 10000;
constexpr std::size_t kCacheMaxWordBytes = 256;

[[noreturn]] void unsupported(const std::string& what) {
  throw std::runtime_error("tokenizer: unsupported tokenizer.json: " + what);
}

bool bool_or(const json::Value& obj, std::string_view key, bool fallback) {
  const json::Value* v = obj.find(key);
  return v == nullptr || v->is_null() ? fallback : v->as_bool();
}

bool null_or_empty(const json::Value& obj, std::string_view key) {
  const json::Value* v = obj.find(key);
  return v == nullptr || v->is_null() || (v->is_string() && v->as_string().empty());
}

void check_pipeline(const json::Value& root) {
  const json::Value* norm = root.find("normalizer");
  if (norm == nullptr || !norm->is_object() || norm->at("type").as_string() != "NFC") {
    unsupported("normalizer must be NFC");
  }
  const json::Value& pre = root.at("pre_tokenizer");
  if (pre.at("type").as_string() != "Sequence") unsupported("pre_tokenizer must be a Sequence");
  const json::Array& steps = pre.at("pretokenizers").as_array();
  if (steps.size() != 2) unsupported("pre_tokenizer must be Split + ByteLevel");
  const json::Value& split = steps[0];
  if (split.at("type").as_string() != "Split" || split.at("behavior").as_string() != "Isolated" ||
      bool_or(split, "invert", false)) {
    unsupported("first pre_tokenizer must be an Isolated, non-inverted Split");
  }
  const json::Value* regex = split.at("pattern").find("Regex");
  if (regex == nullptr || regex->as_string() != kQwen2Pattern) unsupported("Split regex is not Qwen2's");
  const json::Value& byte_level = steps[1];
  if (byte_level.at("type").as_string() != "ByteLevel" || bool_or(byte_level, "add_prefix_space", true) ||
      bool_or(byte_level, "use_regex", true)) {
    unsupported("second pre_tokenizer must be ByteLevel without prefix space or regex");
  }
  // A ByteLevel post-processor adds no tokens, so add_special_tokens=True encodes like False.
  const json::Value* post = root.find("post_processor");
  if (post != nullptr && !post->is_null() && post->at("type").as_string() != "ByteLevel") {
    unsupported("post_processor must be ByteLevel or null");
  }
  const json::Value& model = root.at("model");
  if (model.at("type").as_string() != "BPE") unsupported("model must be BPE");
  if (const json::Value* d = model.find("dropout"); d != nullptr && !d->is_null()) unsupported("BPE dropout");
  if (bool_or(model, "byte_fallback", false)) unsupported("BPE byte_fallback");
  if (bool_or(model, "ignore_merges", false)) unsupported("BPE ignore_merges");
  if (!null_or_empty(model, "continuing_subword_prefix") || !null_or_empty(model, "end_of_word_suffix")) {
    unsupported("BPE subword prefix/suffix");
  }
}

// GPT-2 byte-to-unicode table: printable Latin-1 bytes map to themselves, the rest to U+0100+n.
std::array<char32_t, 256> byte_level_alphabet() {
  std::array<char32_t, 256> map{};
  char32_t next = 256;
  for (unsigned b = 0; b < 256; ++b) {
    bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    map[b] = printable ? static_cast<char32_t>(b) : next++;
  }
  return map;
}

// Open-addressing map (left id, right id) -> (rank, merged id), ~8 MB for Qwen's 247k merges.
class MergeMap {
 public:
  struct Entry {
    std::uint64_t key;
    std::int32_t rank;
    std::int32_t id;
  };

  explicit MergeMap(std::size_t n) {
    std::size_t cap = 16;
    while (cap < 2 * n) cap <<= 1;
    slots_.assign(cap, Entry{kEmpty, 0, 0});
    mask_ = cap - 1;
  }

  // Later merges overwrite earlier ones, like collecting the merge list into a Rust HashMap.
  void insert(std::int32_t a, std::int32_t b, std::int32_t rank, std::int32_t id) {
    std::uint64_t key = pack(a, b);
    std::size_t i = slot(key);
    while (slots_[i].key != kEmpty && slots_[i].key != key) i = (i + 1) & mask_;
    slots_[i] = Entry{key, rank, id};
  }

  const Entry* find(std::int32_t a, std::int32_t b) const {
    std::uint64_t key = pack(a, b);
    for (std::size_t i = slot(key);; i = (i + 1) & mask_) {
      if (slots_[i].key == key) return &slots_[i];
      if (slots_[i].key == kEmpty) return nullptr;
    }
  }

 private:
  static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};
  static std::uint64_t pack(std::int32_t a, std::int32_t b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
  }
  std::size_t slot(std::uint64_t key) const {
    return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ull) >> 32) & mask_;
  }
  std::vector<Entry> slots_;
  std::size_t mask_ = 0;
};

// Leftmost-longest literal matcher over the added tokens of one kind (normalized or not); the
// `tokenizers` library uses an Aho-Corasick automaton with the same match semantics.
class TokenMatcher {
 public:
  struct Match {
    std::size_t start = std::string_view::npos;
    std::size_t end = 0;
    std::size_t token = 0;  // index into Impl::added
  };

  void add(std::string pattern, std::size_t token) {
    if (pattern.empty()) return;
    by_first_[static_cast<unsigned char>(pattern[0])].push_back(patterns_.size());
    patterns_.push_back({std::move(pattern), token});
  }
  bool empty() const { return patterns_.empty(); }

  Match next(std::string_view s, std::size_t from) const {
    for (std::size_t pos = from; pos < s.size(); ++pos) {
      Match best;
      std::size_t best_len = 0;
      for (std::size_t p : by_first_[static_cast<unsigned char>(s[pos])]) {
        const std::string& text = patterns_[p].first;
        if (text.size() > best_len && s.compare(pos, text.size(), text) == 0) {
          best = Match{pos, pos + text.size(), patterns_[p].second};
          best_len = text.size();
        }
      }
      if (best_len > 0) return best;
    }
    return Match{};
  }

 private:
  std::vector<std::pair<std::string, std::size_t>> patterns_;
  std::array<std::vector<std::size_t>, 256> by_first_;
};

struct Piece {
  std::size_t begin, end;
  std::int32_t id;  // -1: plain text
};

struct StringHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

std::size_t prev_char_start(std::string_view s, std::size_t pos) {
  do --pos;
  while (pos > 0 && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80);
  return pos;
}

bool is_space(char32_t cp) { return (regex_class(cp) & kClassS) != 0; }

// Qwen2 pre-tokenizer: splits `s` exactly like the Split regex kQwen2Pattern under Oniguruma's
// leftmost-first alternation, calling emit(piece) for each match. Every position matches some
// alternative, so the matches tile the input.
template <typename Emit>
void split_qwen2(std::string_view s, std::vector<char32_t>& cps, std::vector<std::uint32_t>& offs,
                 std::vector<std::uint8_t>& cls, Emit&& emit) {
  cps.clear();
  offs.clear();
  cls.clear();
  for (std::size_t i = 0; i < s.size();) {
    offs.push_back(static_cast<std::uint32_t>(i));
    char32_t cp = next_code_point(s, i);
    cps.push_back(cp);
    cls.push_back(regex_class(cp));
  }
  offs.push_back(static_cast<std::uint32_t>(s.size()));
  const std::size_t n = cps.size();
  auto is = [&](std::size_t k, std::uint8_t c) { return k < n && (cls[k] & c) != 0; };
  auto other = [&](std::size_t k) { return k < n && cls[k] == 0; };  // [^\s\p{L}\p{N}]
  auto crlf = [&](std::size_t k) { return k < n && (cps[k] == '\r' || cps[k] == '\n'); };
  // (?i:...) folding of the contraction letters: ASCII case, plus U+017F LONG S -> s.
  auto folded = [&](std::size_t k) -> char32_t {
    if (k >= n) return 0;
    char32_t c = cps[k];
    if (c == 0x17F) return 's';
    return c >= 'A' && c <= 'Z' ? c + 32 : c;
  };
  auto run_end = [&](std::size_t k, auto pred) {
    while (k < n && pred(k)) ++k;
    return k;
  };

  for (std::size_t i = 0; i < n;) {
    std::size_t j = 0;
    char32_t c1 = folded(i + 1), c2 = folded(i + 2);
    if (cps[i] == '\'' && (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')) {
      j = i + 2;  // 's 't 'm 'd
    } else if (cps[i] == '\'' && ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))) {
      j = i + 3;  // 're 've 'll
    } else if (is(i, kClassL)) {
      j = run_end(i, [&](std::size_t k) { return is(k, kClassL); });  // \p{L}+
    } else if (!crlf(i) && !is(i, kClassN) && is(i + 1, kClassL)) {
      j = run_end(i + 1, [&](std::size_t k) { return is(k, kClassL); });  // [^\r\n\p{L}\p{N}]\p{L}+
    } else if (is(i, kClassN)) {
      j = i + 1;  // \p{N}
    } else if (other(i) || (cps[i] == ' ' && other(i + 1))) {
      std::size_t k = other(i) ? i : i + 1;  //  ?[^\s\p{L}\p{N}]+[\r\n]*
      j = run_end(run_end(k, other), crlf);
    } else {
      // Whitespace: \s*[\r\n]+ backtracks to the last CR/LF of the run; else \s+(?!\S) leaves
      // the run's last character for the next token unless the run ends the text; else \s+.
      std::size_t r = run_end(i, [&](std::size_t k) { return is(k, kClassS); });
      std::size_t last_crlf = n;
      for (std::size_t k = i; k < r; ++k) {
        if (crlf(k)) last_crlf = k;
      }
      if (last_crlf != n) {
        j = last_crlf + 1;
      } else if (r == n || r - i == 1) {
        j = r;
      } else {
        j = r - 1;
      }
    }
    emit(s.substr(offs[i], offs[j] - offs[i]));
    i = j;
  }
}

}  // namespace

struct Tokenizer::Impl {
  explicit Impl(std::size_t n_merges) : merges(n_merges) {}

  MergeMap merges;
  std::array<std::int32_t, 256> byte_ids{};
  std::vector<AddedToken> added;
  TokenMatcher raw_matcher;         // normalized=false: matched in the raw text
  TokenMatcher normalized_matcher;  // normalized=true: matched after NFC
  std::size_t vocab_size = 0;

  mutable std::mutex cache_mu;
  mutable std::unordered_map<std::string, std::vector<std::int32_t>, StringHash, std::equal_to<>> cache;

  // AddedVocabulary::find_matches, including its single_word / lstrip / rstrip handling.
  std::vector<Piece> find_matches(std::string_view s, const TokenMatcher& m) const {
    std::vector<Piece> out;
    if (s.empty()) {
      out.push_back({0, 0, -1});
      return out;
    }
    std::size_t start_offset = 0;
    for (TokenMatcher::Match mt = m.next(s, 0); mt.start != std::string_view::npos; mt = m.next(s, mt.end)) {
      const AddedToken& tok = added[mt.token];
      std::size_t start = mt.start, stop = mt.end;
      if (tok.single_word) {
        bool start_space = start == 0 || [&] {
          std::size_t p = prev_char_start(s, start);
          return !is_word_char(next_code_point(s, p));
        }();
        bool stop_space = stop == s.size() || [&] {
          std::size_t p = stop;
          return !is_word_char(next_code_point(s, p));
        }();
        if (!start_space || !stop_space) continue;
      }
      if (tok.lstrip) {
        std::size_t new_start = start;
        while (new_start > 0) {
          std::size_t p = prev_char_start(s, new_start), q = p;
          if (!is_space(next_code_point(s, q))) break;
          new_start = p;
        }
        start = std::max(new_start, start_offset);
      }
      if (tok.rstrip) {
        while (stop < s.size()) {
          std::size_t q = stop;
          if (!is_space(next_code_point(s, q))) break;
          stop = q;
        }
      }
      if (start > start_offset) out.push_back({start_offset, start, -1});
      out.push_back({start, stop, tok.id});
      start_offset = stop;
    }
    if (start_offset != s.size()) out.push_back({start_offset, s.size(), -1});
    return out;
  }

  // Word::merge_all of the `tokenizers` BPE model: repeatedly apply the lowest-rank merge (then
  // leftmost position) via a priority queue with lazy invalidation.
  void bpe(std::string_view word, std::vector<std::int32_t>& out) const {
    {
      std::lock_guard lock(cache_mu);
      if (auto it = cache.find(word); it != cache.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
      }
    }
    struct Symbol {
      std::int32_t id, prev, next;
      bool removed;
    };
    struct Merge {
      std::int32_t rank, pos, id;
      bool operator<(const Merge& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    const auto n = static_cast<std::int32_t>(word.size());
    std::vector<Symbol> sym(word.size());
    for (std::int32_t i = 0; i < n; ++i) {
      sym[i] = {byte_ids[static_cast<unsigned char>(word[i])], i - 1, i + 1 < n ? i + 1 : -1, false};
    }
    std::vector<Merge> heap;
    for (std::int32_t i = 0; i + 1 < n; ++i) {
      if (const MergeMap::Entry* m = merges.find(sym[i].id, sym[i + 1].id)) heap.push_back({m->rank, i, m->id});
    }
    std::make_heap(heap.begin(), heap.end());
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end());
      Merge top = heap.back();
      heap.pop_back();
      Symbol& cur = sym[top.pos];
      if (cur.removed || cur.next == -1) continue;
      const Symbol right = sym[cur.next];
      const MergeMap::Entry* m = merges.find(cur.id, right.id);
      if (m == nullptr || m->id != top.id) continue;  // stale entry
      sym[cur.next].removed = true;
      cur.id = top.id;
      cur.next = right.next;
      if (right.next != -1) sym[right.next].prev = top.pos;
      if (cur.prev != -1) {
        if (const MergeMap::Entry* pm = merges.find(sym[cur.prev].id, cur.id)) {
          heap.push_back({pm->rank, cur.prev, pm->id});
          std::push_heap(heap.begin(), heap.end());
        }
      }
      if (cur.next != -1) {
        if (const MergeMap::Entry* nm = merges.find(cur.id, sym[cur.next].id)) {
          heap.push_back({nm->rank, top.pos, nm->id});
          std::push_heap(heap.begin(), heap.end());
        }
      }
    }
    std::vector<std::int32_t> ids;
    for (const Symbol& s : sym) {
      if (!s.removed) ids.push_back(s.id);
    }
    out.insert(out.end(), ids.begin(), ids.end());
    std::lock_guard lock(cache_mu);
    if (word.size() < kCacheMaxWordBytes && cache.size() < kCacheCapacity) cache.emplace(std::string(word), std::move(ids));
  }

  void encode_plain(std::string_view s, std::vector<std::int32_t>& out) const {
    std::vector<char32_t> cps;
    std::vector<std::uint32_t> offs;
    std::vector<std::uint8_t> cls;
    split_qwen2(s, cps, offs, cls, [&](std::string_view piece) { bpe(piece, out); });
  }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::load(const std::filesystem::path& tokenizer_json) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(tokenizer_json, ec);
  if (ec) throw std::runtime_error("tokenizer: cannot open " + util::path_utf8(tokenizer_json));
  if (size > kMaxTokenizerBytes) {
    throw std::runtime_error("tokenizer: " + util::path_utf8(tokenizer_json) + " is " + std::to_string(size) +
                             " bytes, over the limit of " + std::to_string(kMaxTokenizerBytes));
  }
  std::ifstream in(tokenizer_json, std::ios::binary);
  if (!in) throw std::runtime_error("tokenizer: cannot open " + util::path_utf8(tokenizer_json));
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) throw std::runtime_error("tokenizer: cannot read " + util::path_utf8(tokenizer_json));
  if (text.size() > kMaxTokenizerBytes) throw std::runtime_error("tokenizer: " + util::path_utf8(tokenizer_json) + " grew past the size limit while reading");
  try {
    return from_json(json::parse(text, {.max_values = kMaxTokenizerValues}));
  } catch (const json::ParseError& e) {
    throw std::runtime_error("tokenizer: " + util::path_utf8(tokenizer_json) + ": " + e.what());
  }
}

Tokenizer Tokenizer::from_json(const json::Value& root) {
  check_pipeline(root);
  const json::Value& model = root.at("model");
  const json::Object& vocab = model.at("vocab").as_object();
  const json::Array& merges = model.at("merges").as_array();

  std::unordered_map<std::string_view, std::int32_t> ids;
  ids.reserve(vocab.size());
  std::int64_t max_id = -1;
  for (const auto& [token, id] : vocab) {
    std::int64_t v = id.as_int();
    if (v < 0 || v > std::numeric_limits<std::int32_t>::max()) unsupported("vocab id out of range");
    ids.emplace(token, static_cast<std::int32_t>(v));
    max_id = std::max(max_id, v);
  }
  auto id_of = [&](std::string_view token) {
    auto it = ids.find(token);
    if (it == ids.end()) unsupported("merge references unknown token '" + std::string(token) + "'");
    return it->second;
  };

  auto impl = std::make_unique<Impl>(merges.size());
  std::string joined;
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    std::string_view a, b;
    if (merges[rank].is_string()) {  // legacy "a b" form
      std::string_view m = merges[rank].as_string();
      std::size_t sp = m.find(' ');
      if (sp == std::string_view::npos || m.find(' ', sp + 1) != std::string_view::npos) unsupported("bad merge");
      a = m.substr(0, sp), b = m.substr(sp + 1);
    } else {
      const json::Array& pair = merges[rank].as_array();
      if (pair.size() != 2) unsupported("bad merge");
      a = pair[0].as_string(), b = pair[1].as_string();
    }
    joined.assign(a).append(b);
    impl->merges.insert(id_of(a), id_of(b), static_cast<std::int32_t>(rank), id_of(joined));
  }

  const std::array<char32_t, 256> alphabet = byte_level_alphabet();
  for (std::size_t b = 0; b < 256; ++b) {
    std::string ch;
    append_utf8(ch, alphabet[b]);
    auto it = ids.find(ch);
    if (it == ids.end()) unsupported("byte-level symbol for byte " + std::to_string(b) + " missing from vocab");
    impl->byte_ids[b] = it->second;
  }

  if (const json::Value* added = root.find("added_tokens"); added != nullptr) {
    for (const json::Value& t : added->as_array()) {
      AddedToken tok;
      tok.content = t.at("content").as_string();
      validate_utf8(tok.content);
      const std::int64_t id = t.at("id").as_int();
      // Same range as vocab ids; encode() also relies on ids being >= 0 (-1 marks plain text).
      if (id < 0 || id > std::numeric_limits<std::int32_t>::max()) unsupported("added token id out of range");
      tok.id = static_cast<std::int32_t>(id);
      tok.single_word = bool_or(t, "single_word", false);
      tok.lstrip = bool_or(t, "lstrip", false);
      tok.rstrip = bool_or(t, "rstrip", false);
      tok.normalized = bool_or(t, "normalized", true);
      tok.special = bool_or(t, "special", false);
      max_id = std::max<std::int64_t>(max_id, tok.id);
      std::size_t index = impl->added.size();
      if (tok.normalized) {
        impl->normalized_matcher.add(nfc(tok.content), index);
      } else {
        impl->raw_matcher.add(tok.content, index);
      }
      impl->added.push_back(std::move(tok));
    }
  }
  impl->vocab_size = static_cast<std::size_t>(max_id + 1);
  return Tokenizer(std::move(impl));
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text) const {
  validate_utf8(text);
  std::vector<std::int32_t> out;
  out.reserve(text.size() / 2);
  for (const Piece& p : impl_->find_matches(text, impl_->raw_matcher)) {
    if (p.id >= 0) {
      out.push_back(p.id);
      continue;
    }
    std::string normalized = nfc(text.substr(p.begin, p.end - p.begin));
    if (impl_->normalized_matcher.empty()) {
      impl_->encode_plain(normalized, out);
      continue;
    }
    for (const Piece& q : impl_->find_matches(normalized, impl_->normalized_matcher)) {
      if (q.id >= 0) {
        out.push_back(q.id);
      } else {
        impl_->encode_plain(std::string_view(normalized).substr(q.begin, q.end - q.begin), out);
      }
    }
  }
  return out;
}

std::size_t Tokenizer::vocab_size() const { return impl_->vocab_size; }

const std::vector<AddedToken>& Tokenizer::added_tokens() const { return impl_->added; }

std::vector<std::string_view> qwen2_pretokenize(std::string_view text) {
  validate_utf8(text);
  std::vector<std::string_view> pieces;
  std::vector<char32_t> cps;
  std::vector<std::uint32_t> offs;
  std::vector<std::uint8_t> cls;
  split_qwen2(text, cps, offs, cls, [&](std::string_view piece) { pieces.push_back(piece); });
  return pieces;
}

std::string_view python_strip(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t next = begin;
    if (!is_python_space(next_code_point(s, next))) break;
    begin = next;
  }
  std::size_t end = s.size();
  while (end > begin) {
    std::size_t p = prev_char_start(s, end), q = p;
    if (!is_python_space(next_code_point(s, q))) break;
    end = p;
  }
  return s.substr(begin, end - begin);
}

std::string format_moss_prompt(std::string_view prompt) {
  validate_utf8(prompt);
  std::string_view content = python_strip(prompt);
  // The template raises "No user query found in messages." when the only user turn looks like a
  // tool response.
  if (content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
    throw std::invalid_argument("tokenizer: chat template rejects a prompt wrapped in <tool_response> tags");
  }
  std::string out;
  out.reserve(content.size() + 64);
  out.append("<|im_start|>user\n").append(content).append("<|im_end|>\n<|im_start|>assistant\n<think>\n");
  return out;
}

std::vector<std::int32_t> encode_moss_prompt(const Tokenizer& tokenizer, std::string_view prompt,
                                             std::size_t max_length) {
  std::vector<std::int32_t> ids = tokenizer.encode(format_moss_prompt(prompt));
  if (ids.size() > max_length) ids.resize(max_length);
  return ids;
}

}  // namespace mi::tok
