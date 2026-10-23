#!/usr/bin/env python3
"""Release version of a tag, checked against include/mossinfer/mossinfer.h and the repository's tags (release.yml).

  release_version.py [--tag TAG] [--sha SHA] [--header PATH]   prints version=, tag=, is_rc=, archive_version=,
                                                                previous_tag= lines
  release_version.py --self-test                                runs the accepted / rejected tag tests

TAG is vX.Y.Z (a release) or vX.Y.Z-rcN with N >= 1 (a release candidate, published as a pre-release); X.Y.Z must
equal the header's MI_VERSION_MAJOR.MINOR.PATCH, and an rc is refused once vX.Y.Z exists. Without --tag (a dry run
from a branch) the would-be tag is v<header version>. The archives are named by CPack from the header, so
archive_version is X.Y.Z for an rc too. previous_tag (final tags only, else empty) is the highest earlier final tag
reachable from SHA, the base of the generated release notes. Tags are read with git from the current directory.
"""
import argparse
import re
import subprocess
import sys

TAG_RE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-rc([1-9][0-9]*))?")


def parse_tag(tag):
    """((X, Y, Z), rc number or None) for a well-formed tag, else None."""
    m = TAG_RE.fullmatch(tag)
    if not m:
        return None
    return tuple(int(g) for g in m.group(1, 2, 3)), (int(m.group(4)) if m.group(4) else None)


def header_version(text):
    parts = []
    for p in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"^#define MI_VERSION_{p} +([0-9]+)", text, re.M)
        if not m:
            raise ValueError(f"MI_VERSION_{p} not found in the header")
        parts.append(m.group(1))
    return ".".join(parts)


def previous_final(version, merged):
    """First final tag in MERGED (newest first) older than VERSION, or ""."""
    current = tuple(int(p) for p in version.split("."))
    for t in merged:
        p = parse_tag(t)
        if p and p[1] is None and p[0] < current:
            return t
    return ""


def release_info(tag, header, all_tags, merged):
    """(outputs, warnings) for TAG (None: a dry run from a branch); ValueError for a tag release.yml refuses.

    ALL_TAGS: every tag of the repository; MERGED: the tags reachable from the commit, newest version first."""
    final = f"v{header}"
    if tag is None:
        warnings = [f"{final} is already released; bump MI_VERSION_* before the next release"] \
            if final in all_tags else []
        return {"version": header, "tag": final, "is_rc": "false", "archive_version": header,
                "previous_tag": previous_final(header, merged)}, warnings
    p = parse_tag(tag)
    if not p:
        raise ValueError(f"tag {tag!r} is neither vX.Y.Z nor vX.Y.Z-rcN (N >= 1, no leading zeros)")
    version = ".".join(str(n) for n in p[0])
    if version != header:
        raise ValueError(f"tag {tag} names version {version}, but include/mossinfer/mossinfer.h says {header}")
    is_rc = p[1] is not None
    if is_rc and final in all_tags:
        raise ValueError(f"{final} is already released; bump MI_VERSION_* first")
    return {"version": version, "tag": tag, "is_rc": "true" if is_rc else "false", "archive_version": version,
            "previous_tag": "" if is_rc else previous_final(version, merged)}, []


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def self_test():
    tags = ["v1.0.0-rc1", "v1.0.0", "v1.0.1-rc1", "v1.0.1", "v1.1.0-rc1", "v1.1.0-rc2", "vnext", "v1.2"]
    merged = ["v1.1.0-rc2", "v1.1.0-rc1", "v1.0.1", "v1.0.1-rc1", "v1.0.0", "v1.0.0-rc1"]
    accepted = {  # tag, header -> is_rc, previous_tag
        ("v1.0.0", "1.0.0"): ("false", ""),
        ("v1.0.0-rc1", "1.0.0"): ("true", ""),
        ("v1.0.0-rc12", "1.0.0"): ("true", ""),
        ("v1.0.1", "1.0.1"): ("false", "v1.0.0"),
        ("v1.1.0", "1.1.0"): ("false", "v1.0.1"),
        ("v1.1.0-rc3", "1.1.0"): ("true", ""),
        ("v2.0.0", "2.0.0"): ("false", "v1.0.1"),
        ("v10.20.30-rc3", "10.20.30"): ("true", ""),
    }
    for (tag, header), (is_rc, prev) in accepted.items():
        all_tags = tags if header != "1.0.0" else []
        info, warnings = release_info(tag, header, all_tags, merged if header != "1.0.0" else [])
        check(info == {"version": header, "tag": tag, "is_rc": is_rc, "archive_version": header,
                       "previous_tag": prev} and not warnings, f"{tag}: {info} {warnings}")
    info, warnings = release_info(None, "1.1.0", tags, merged)
    check(info == {"version": "1.1.0", "tag": "v1.1.0", "is_rc": "false", "archive_version": "1.1.0",
                   "previous_tag": "v1.0.1"} and not warnings, f"branch: {info} {warnings}")
    info, warnings = release_info(None, "1.0.1", tags, merged)
    check(info["previous_tag"] == "v1.0.0" and len(warnings) == 1, f"branch, released: {info} {warnings}")
    rejected = ["v1.0", "v1", "1.0.0", "v1.0.0-beta", "v1.0.0-rc0", "v1.0.0-rc", "v1.0.0-rc01", "v1.0.0rc1",
                "v1.0.0-RC1", "v1.0.0-rc1-hotfix", "v01.0.0", "v1.00.0", "v1.0.0.0", "v1.0.0+build", "v1.0.0\n",
                " v1.0.0", "v1.0.0-rc1\n", "V1.0.0", "v١.0.0", "v1.0.1", "v1.1.0-rc1", "v2.0.0"]
    for tag in rejected:
        try:
            release_info(tag, "1.0.0", [], [])
        except ValueError:
            continue
        raise AssertionError(f"accepted {tag!r}")
    for tag in ("v1.0.1-rc2", "v1.0.0-rc2"):  # an rc of a released version
        try:
            release_info(tag, tag[1:6], tags, merged)
        except ValueError as e:
            check("already released" in str(e), str(e))
            continue
        raise AssertionError(f"accepted {tag!r} after its final")
    check(header_version("#define MI_VERSION_MAJOR 1\n#define MI_VERSION_MINOR 10\n#define MI_VERSION_PATCH 3\n")
          == "1.10.3", "header parse")
    try:
        header_version("#define MI_VERSION_MAJOR 1\n")
        raise AssertionError("header without MINOR accepted")
    except ValueError:
        pass
    print(f"self-test: {len(accepted) + 2} accepted, {len(rejected) + 2} rejected, header parse ok")


def git_tags(*args):
    out = subprocess.run(["git", "tag", "--list", "v[0-9]*", *args], check=True, capture_output=True, text=True)
    return out.stdout.split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="the pushed tag (omit for a dry run from a branch)")
    ap.add_argument("--sha", default="HEAD", help="the commit being released")
    ap.add_argument("--header", default="include/mossinfer/mossinfer.h")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        with open(args.header, encoding="utf-8") as f:
            header = header_version(f.read())
        info, warnings = release_info(args.tag, header, git_tags(),
                                      git_tags("--merged", args.sha, "--sort=-v:refname"))
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"::warning::{w}", file=sys.stderr)
    for k, v in info.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
