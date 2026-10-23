#!/usr/bin/env python3
"""Release version of a tag, checked against include/mossinfer/mossinfer.h (called by release.yml).

  release_version.py [--tag TAG] [--header PATH]   prints version=, tag=, is_rc=, archive_version= lines
  release_version.py --self-test                    runs the accepted / rejected tag tests

TAG is vX.Y.Z (a release) or vX.Y.Z-rcN with N >= 1 (a release candidate, published as a pre-release); X.Y.Z must
equal the header's MI_VERSION_MAJOR.MINOR.PATCH. Without --tag (a dry run from a branch) the would-be tag is
v<header version>. The archives are named by CPack from the header, so archive_version is X.Y.Z for an rc too.
"""
import argparse
import re
import sys

TAG_RE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-rc([1-9][0-9]*))?")


def header_version(text):
    parts = []
    for p in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"^#define MI_VERSION_{p} +([0-9]+)", text, re.M)
        if not m:
            raise ValueError(f"MI_VERSION_{p} not found in the header")
        parts.append(m.group(1))
    return ".".join(parts)


def release_info(tag, header):
    """Release outputs for TAG (None: no tag, a dry run from a branch); ValueError for a tag release.yml rejects."""
    if tag is None:
        return {"version": header, "tag": f"v{header}", "is_rc": "false", "archive_version": header}
    m = TAG_RE.fullmatch(tag)
    if not m:
        raise ValueError(f"tag {tag!r} is neither vX.Y.Z nor vX.Y.Z-rcN (N >= 1, no leading zeros)")
    version = ".".join(m.group(1, 2, 3))
    if version != header:
        raise ValueError(f"tag {tag} names version {version}, but include/mossinfer/mossinfer.h says {header}")
    is_rc = m.group(4) is not None
    return {"version": version, "tag": tag, "is_rc": "true" if is_rc else "false", "archive_version": version}


def self_test():
    accepted = {
        "v1.0.0": ("1.0.0", "false"),
        "v1.0.0-rc1": ("1.0.0", "true"),
        "v1.0.0-rc2": ("1.0.0", "true"),
        "v1.0.0-rc10": ("1.0.0", "true"),
    }
    for tag, (version, is_rc) in accepted.items():
        info = release_info(tag, "1.0.0")
        assert info == {"version": version, "tag": tag, "is_rc": is_rc, "archive_version": version}, (tag, info)
    assert release_info("v10.20.30-rc3", "10.20.30")["is_rc"] == "true"
    assert release_info(None, "1.1.0") == {"version": "1.1.0", "tag": "v1.1.0", "is_rc": "false",
                                           "archive_version": "1.1.0"}
    rejected = ["v1.0", "v1", "1.0.0", "v1.0.0-beta", "v1.0.0-rc0", "v1.0.0-rc", "v1.0.0-rc01", "v1.0.0rc1",
                "v1.0.0-RC1", "v1.0.0-rc1-hotfix", "v01.0.0", "v1.00.0", "v1.0.0.0", "v1.0.0+build", "v1.0.0\n",
                " v1.0.0", "v1.0.0-rc1\n", "V1.0.0", "v١.0.0", "v1.0.1", "v1.1.0-rc1", "v2.0.0"]
    for tag in rejected:
        try:
            release_info(tag, "1.0.0")
        except ValueError:
            continue
        raise AssertionError(f"accepted {tag!r}")
    assert header_version("#define MI_VERSION_MAJOR 1\n#define MI_VERSION_MINOR 10\n#define MI_VERSION_PATCH 3\n") \
        == "1.10.3"
    try:
        header_version("#define MI_VERSION_MAJOR 1\n")
        raise AssertionError("header without MINOR accepted")
    except ValueError:
        pass
    print(f"self-test: {len(accepted) + 2} accepted, {len(rejected)} rejected, header parse ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="the pushed tag (omit for a dry run from a branch)")
    ap.add_argument("--header", default="include/mossinfer/mossinfer.h")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        with open(args.header, encoding="utf-8") as f:
            info = release_info(args.tag, header_version(f.read()))
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
