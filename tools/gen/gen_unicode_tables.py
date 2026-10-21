#!/usr/bin/env python3
"""Generates src/tokenizer/unicode_tables.inc for the Qwen tokenizer port.

Every table is measured from the installed HF `tokenizers` library itself rather than taken from
Python's `unicodedata`, because the Rust crate bundles three different Unicode snapshots:

  * the Split pre-tokenizer regex runs on Oniguruma, whose \\p{L} / \\p{N} / \\s tables are newer
    than Python 3.12's unicodedata (15.0);
  * the NFC normalizer comes from `unicode-normalization-alignments`, whose combining classes and
    compositions are much older (Unicode 9-era: e.g. U+11938 does not decompose, and ~100 marks
    added since then have combining class 0 there);
  * added-token `single_word` / `lstrip` / `rstrip` use Rust `char` predicates.

`unicodedata` only supplies candidates (single-step decompositions, combining classes); each one is
confirmed or corrected by probing `tokenizers`. Differences from `unicodedata` are printed so a
library upgrade that moves a table is visible in the diff of this script's output.

Dev-only. Run with the reference venv (CPU only):
  CUDA_VISIBLE_DEVICES= tools/reference/.venv/bin/python tools/gen/gen_unicode_tables.py
"""

import sys
import unicodedata as ud
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers

OUT = Path(__file__).resolve().parents[2] / "src/tokenizer/unicode_tables.inc"
CPS = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000]
CHUNK = 4096

kL, kN, kS = 1, 2, 4  # must match unicode.h


def onig_set(pattern):
    """Code points that the Oniguruma pattern matches as a single character."""
    split = pre_tokenizers.Split(Regex(pattern), "removed")
    out = set()
    for i in range(0, len(CPS), CHUNK):
        chunk = CPS[i:i + CHUNK]
        covered = bytearray(len(chunk))
        # "removed" returns the unmatched spans (char offsets); everything else matched.
        for _, (a, b) in split.pre_tokenize_str("".join(map(chr, chunk))):
            covered[a:b] = b"\x01" * (b - a)
        out.update(c for c, cv in zip(chunk, covered) if not cv)
    return out


def ranges(cps):
    out = []
    for c in sorted(cps):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def report(name, ours, theirs):
    extra, missing = sorted(ours - theirs), sorted(theirs - ours)
    fmt = lambda xs: " ".join(f"{c:04X}" for c in xs[:12]) + (" ..." if len(xs) > 12 else "")
    print(f"  {name}: {len(ours)} cps; +{len(extra)} vs unicodedata [{fmt(extra)}]; "
          f"-{len(missing)} [{fmt(missing)}]")


def regex_classes():
    L, N, S = onig_set(r"\p{L}"), onig_set(r"\p{N}"), onig_set(r"\s")
    assert not (L & N) and not (L & S) and not (N & S)
    report("onig \\p{L}", L, {c for c in CPS if ud.category(chr(c))[0] == "L"})
    report("onig \\p{N}", N, {c for c in CPS if ud.category(chr(c))[0] == "N"})
    print("  onig \\s:", " ".join(f"{c:04X}" for c in sorted(S)))
    # The case-insensitive contraction alternative (?i:'s|'t|'re|'ve|'m|'ll|'d) is hand-coded in
    # qwen_bpe.cpp: each letter matches its two ASCII cases, and 's' also matches U+017F (long s).
    for letter in "strevmld":
        got = onig_set(f"(?i:{letter})")
        want = {ord(letter), ord(letter.upper())} | ({0x17F} if letter == "s" else set())
        assert got == want, (letter, sorted(map(hex, got)))
    for multi in ("re", "ve", "ll"):
        assert all(c < 0x80 for c in onig_set(f"(?i:{multi})")), multi  # no 1-char folds
    flags = {}
    for bit, cps in ((kL, L), (kN, N), (kS, S)):
        for c in cps:
            flags[c] = flags.get(c, 0) | bit
    out = []
    for c in sorted(flags):
        if out and out[-1][1] == c - 1 and out[-1][2] == flags[c]:
            out[-1][1] = c
        else:
            out.append([c, c, flags[c]])
    return out, S


def nfc_tables():
    nfd, nfc = normalizers.NFD(), normalizers.NFC()
    decomp = {}
    for c in CPS:
        s = chr(c)
        d = nfd.normalize_str(s)
        if 0xAC00 <= c <= 0xD7A3:  # Hangul syllables are algorithmic; check the crate agrees
            assert d == ud.normalize("NFD", s), hex(c)
            continue
        if d != s:
            decomp[c] = [ord(x) for x in d]
    ud_decomp = {c for c in CPS if ud.normalize("NFD", chr(c)) != chr(c) and not 0xAC00 <= c <= 0xD7A3}
    report("crate NFD-changing", set(decomp), ud_decomp)

    # Combining classes. Values are stable once assigned, so the crate's class is either
    # unicodedata's or 0 (character unknown to the crate). Probe reordering against U+0334
    # (ccc 1) and U+0301 (ccc 230), both ancient.
    def swaps_after(c):  # crate ccc(c) > 1
        return nfd.normalize_str(chr(c) + "̴") == "̴" + chr(c)

    def swaps_before(c):  # 0 < crate ccc(c) < 230
        return nfd.normalize_str("́" + chr(c)) == chr(c) + "́"

    ccc = {}
    for c in CPS:
        if c in decomp or 0xAC00 <= c <= 0xD7A3:
            continue  # only fully decomposed characters are ever looked up
        k = ud.combining(chr(c))
        if k == 0:
            assert not swaps_after(c) and not swaps_before(c), f"crate ccc>0 for U+{c:04X}"
            continue
        if swaps_after(c) if k > 1 else swaps_before(c):
            ccc[c] = k
    report("crate ccc>0", set(ccc), {c for c in CPS if ud.combining(chr(c)) and c not in decomp})

    # Compositions: single-step canonical pairs (from unicodedata) of characters the crate
    # decomposes, kept only when the crate recomposes them (i.e. not composition-excluded).
    comps = []
    for c, full in decomp.items():
        parts = ud.decomposition(chr(c)).split()
        if len(parts) != 2 or parts[0].startswith("<"):
            continue
        a, b = (int(p, 16) for p in parts)
        assert nfd.normalize_str(chr(a) + chr(b)) == "".join(map(chr, full)), hex(c)
        if nfc.normalize_str("".join(map(chr, full))) == chr(c):
            comps.append((a, b, c))
    ud_comps = {c for c in CPS if ud.normalize("NFC", ud.normalize("NFD", chr(c))) == chr(c)
                and len(ud.decomposition(chr(c)).split()) == 2
                and not ud.decomposition(chr(c)).startswith("<") and not 0xAC00 <= c <= 0xD7A3}
    report("crate compositions", {c for _, _, c in comps}, ud_comps)
    comps.sort()
    return decomp, ccc, comps


def added_token_predicates(regex_space):
    """Char predicates used by AddedVocabulary for single_word and lstrip/rstrip."""
    def tok(**kw):
        t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        t.add_tokens([AddedToken("<zz>", **kw)])
        return t

    sw = tok(single_word=True)
    before = {c for c, e in zip(CPS, sw.encode_batch([chr(c) + "<zz>" for c in CPS])) if 1 not in e.ids}
    after = {c for c, e in zip(CPS, sw.encode_batch(["<zz>" + chr(c) for c in CPS])) if 1 not in e.ids}
    assert before == after
    report("added-token word chars", before, {c for c in CPS if ud.category(chr(c))[0] in "LN"})
    ls = tok(lstrip=True)
    space = {c for c, e in zip(CPS, ls.encode_batch(["a" + chr(c) + "<zz>" for c in CPS]))
             if e.offsets[-1][0] == 1}
    rs = tok(rstrip=True)
    rspace = {c for c, e in zip(CPS, rs.encode_batch(["<zz>" + chr(c) + "a" for c in CPS]))
              if e.offsets[0][1] == 5}
    # unicode.cpp reuses the regex \s flag for lstrip/rstrip; regenerate differently if this fails.
    assert space == rspace == regex_space
    return ranges(before)


def main():
    print(f"tokenizers {tokenizers.__version__}, Python unicodedata {ud.unidata_version}")
    classes, space = regex_classes()
    decomp, ccc, comps = nfc_tables()
    word = added_token_predicates(space)
    py_space = [c for c in CPS if chr(c).isspace()]
    ccc_ranges = []
    for c in sorted(ccc):
        if ccc_ranges and ccc_ranges[-1][1] == c - 1 and ccc_ranges[-1][2] == ccc[c]:
            ccc_ranges[-1][1] = c
        else:
            ccc_ranges.append([c, c, ccc[c]])

    def rows(items, per_line):
        lines = []
        for i in range(0, len(items), per_line):
            lines.append("    " + " ".join(items[i:i + per_line]))
        return "\n".join(lines)

    pool, index = [], []
    for c in sorted(decomp):
        index.append(f"{{0x{c:X}, {len(pool)}, {len(decomp[c])}}},")
        pool.extend(decomp[c])
    assert len(pool) < 65536 and max(len(v) for v in decomp.values()) <= 4

    text = f"""// Generated by tools/gen/gen_unicode_tables.py from tokenizers {tokenizers.__version__}
// (Python unicodedata {ud.unidata_version} used only for candidates). Do not edit.
// clang-format off

// Oniguruma classes used by the Qwen2 Split regex: kClassL=\\p{{L}}, kClassN=\\p{{N}}, kClassS=\\s.
// The same \\s set is what added tokens' lstrip/rstrip strip (Rust char::is_whitespace).
constexpr ClassRange kClassRanges[] = {{
{rows([f"{{0x{a:X}, 0x{b:X}, {f}}}," for a, b, f in classes], 6)}
}};

// Characters that count as word characters for added tokens with single_word=true.
constexpr CodeRange kWordCharRanges[] = {{
{rows([f"{{0x{a:X}, 0x{b:X}}}," for a, b in word], 8)}
}};

// Python str.isspace() (used by Jinja's |trim, i.e. str.strip()).
constexpr char32_t kPythonSpace[] = {{
{rows([f"0x{c:X}," for c in py_space], 12)}
}};

// NFC data of the tokenizers crate: nonzero canonical combining classes,
constexpr CccRange kCccRanges[] = {{
{rows([f"{{0x{a:X}, 0x{b:X}, {k}}}," for a, b, k in ccc_ranges], 6)}
}};

// full canonical decompositions (Hangul syllables excluded: algorithmic),
constexpr Decomposition kDecompositions[] = {{
{rows(index, 6)}
}};
constexpr char32_t kDecompositionPool[] = {{
{rows([f"0x{c:X}," for c in pool], 12)}
}};

// and primary compositions sorted by (first, second).
constexpr Composition kCompositions[] = {{
{rows([f"{{0x{a:X}, 0x{b:X}, 0x{c:X}}}," for a, b, c in comps], 4)}
}};
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes): {len(classes)} class ranges, {len(word)} word ranges, "
          f"{len(ccc_ranges)} ccc ranges, {len(decomp)} decompositions, {len(comps)} compositions")


if __name__ == "__main__":
    sys.exit(main())
