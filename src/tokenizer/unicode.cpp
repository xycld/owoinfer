#include "tokenizer/unicode.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mi::tok {

namespace {

struct ClassRange {
  char32_t lo, hi;
  std::uint8_t flags;
};
struct CodeRange {
  char32_t lo, hi;
};
struct CccRange {
  char32_t lo, hi;
  std::uint8_t ccc;
};
struct Decomposition {
  char32_t cp;
  std::uint16_t offset;
  std::uint8_t length;
};
struct Composition {
  char32_t first, second, composite;
};

#include "tokenizer/unicode_tables.inc"

// Hangul syllable constants (Unicode §3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

template <typename Range>
const Range* find_range(const Range* begin, const Range* end, char32_t cp) {
  const Range* it = std::upper_bound(begin, end, cp, [](char32_t c, const Range& r) { return c < r.lo; });
  if (it == begin) return nullptr;
  --it;
  return cp <= it->hi ? it : nullptr;
}

std::uint8_t combining_class(char32_t cp) {
  if (cp < 0x300) return 0;
  const CccRange* r = find_range(std::begin(kCccRanges), std::end(kCccRanges), cp);
  return r ? r->ccc : 0;
}

void decompose(char32_t cp, std::vector<char32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    char32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount != 0) out.push_back(kTBase + s % kTCount);
    return;
  }
  const Decomposition* it = std::lower_bound(std::begin(kDecompositions), std::end(kDecompositions), cp,
                                             [](const Decomposition& d, char32_t c) { return d.cp < c; });
  if (it != std::end(kDecompositions) && it->cp == cp) {
    out.insert(out.end(), kDecompositionPool + it->offset, kDecompositionPool + it->offset + it->length);
  } else {
    out.push_back(cp);
  }
}

// Returns the primary composite of (a, b), or 0 if there is none.
char32_t compose(char32_t a, char32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  }
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
      b < kTBase + kTCount) {
    return a + (b - kTBase);
  }
  const Composition* it = std::lower_bound(
      std::begin(kCompositions), std::end(kCompositions), std::pair{a, b},
      [](const Composition& c, const std::pair<char32_t, char32_t>& k) {
        return c.first != k.first ? c.first < k.first : c.second < k.second;
      });
  if (it != std::end(kCompositions) && it->first == a && it->second == b) return it->composite;
  return 0;
}

[[noreturn]] void bad_utf8(std::size_t pos) {
  throw std::invalid_argument("tokenizer: invalid UTF-8 at byte " + std::to_string(pos));
}

}  // namespace

void validate_utf8(std::string_view s) {
  std::size_t i = 0;
  while (i < s.size()) {
    auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
      ++i;
      continue;
    }
    std::size_t len;
    char32_t min;
    char32_t cp;
    if ((b0 & 0xE0) == 0xC0) {
      len = 2, min = 0x80, cp = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3, min = 0x800, cp = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
      len = 4, min = 0x10000, cp = b0 & 0x07u;
    } else {
      bad_utf8(i);
    }
    if (i + len > s.size()) bad_utf8(i);
    for (std::size_t k = 1; k < len; ++k) {
      auto b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) bad_utf8(i);
      cp = (cp << 6) | (b & 0x3Fu);
    }
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) bad_utf8(i);
    i += len;
  }
}

char32_t next_code_point(std::string_view s, std::size_t& i) {
  auto b0 = static_cast<unsigned char>(s[i++]);
  if (b0 < 0x80) return b0;
  std::size_t extra = b0 >= 0xF0 ? 3 : b0 >= 0xE0 ? 2 : 1;
  char32_t cp = b0 & (0x3Fu >> extra);
  for (std::size_t k = 0; k < extra; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3Fu);
  return cp;
}

void append_utf8(std::string& out, char32_t cp) {
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

std::uint8_t regex_class(char32_t cp) {
  if (cp < 0x80) {
    if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return kClassL;
    if (cp >= '0' && cp <= '9') return kClassN;
    return (cp == ' ' || (cp >= '\t' && cp <= '\r')) ? kClassS : 0;
  }
  const ClassRange* r = find_range(std::begin(kClassRanges), std::end(kClassRanges), cp);
  return r ? r->flags : 0;
}

bool is_word_char(char32_t cp) {
  return find_range(std::begin(kWordCharRanges), std::end(kWordCharRanges), cp) != nullptr;
}

bool is_python_space(char32_t cp) {
  return std::find(std::begin(kPythonSpace), std::end(kPythonSpace), cp) != std::end(kPythonSpace);
}

std::string nfc(std::string_view s) {
  // ASCII never decomposes, reorders or composes.
  if (std::all_of(s.begin(), s.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; })) {
    return std::string(s);
  }

  std::vector<char32_t> cps;
  cps.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) decompose(next_code_point(s, i), cps);

  // Canonical ordering: stable sort of every run of non-starters by combining class (an
  // O(n log n) sort, since hostile input can hold arbitrarily long runs of marks).
  std::vector<std::uint8_t> ccc(cps.size());
  for (std::size_t i = 0; i < cps.size(); ++i) ccc[i] = combining_class(cps[i]);
  std::vector<std::pair<std::uint8_t, char32_t>> run;
  for (std::size_t i = 0; i < cps.size();) {
    if (ccc[i] == 0) {
      ++i;
      continue;
    }
    std::size_t end = i;
    while (end < cps.size() && ccc[end] != 0) ++end;
    if (end - i > 1) {
      run.clear();
      for (std::size_t k = i; k < end; ++k) run.emplace_back(ccc[k], cps[k]);
      std::stable_sort(run.begin(), run.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      for (std::size_t k = i; k < end; ++k) std::tie(ccc[k], cps[k]) = run[k - i];
    }
    i = end;
  }

  // Canonical composition (UAX #15): a character combines with the last starter unless blocked
  // by an intervening character of class 0 or of class >= its own.
  std::string out;
  out.reserve(s.size());
  std::vector<char32_t> composed;
  composed.reserve(cps.size());
  std::size_t starter = SIZE_MAX;
  int last_class = -1;  // class of the last character kept after the starter; -1: adjacent
  for (std::size_t i = 0; i < cps.size(); ++i) {
    char32_t c = cps[i];
    int cc = ccc[i];
    if (starter != SIZE_MAX && (last_class == -1 || last_class < cc)) {
      if (char32_t comp = compose(composed[starter], c)) {
        composed[starter] = comp;
        continue;
      }
    }
    if (cc == 0) {
      starter = composed.size();
      last_class = -1;
    } else {
      last_class = cc;
    }
    composed.push_back(c);
  }
  for (char32_t c : composed) append_utf8(out, c);
  return out;
}

}  // namespace mi::tok
