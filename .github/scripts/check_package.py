#!/usr/bin/env python3
"""Checks an installed MossInfer tree or a CPack archive (CI and local).

  AC1   no torch / c10 / python among the dependencies of any shipped binary (readelf+ldd / dumpbin /dependents)
  ABI   the shared library exports only mi_* symbols (nm -D / dumpbin /exports)
  AC17  engine library <= 150 MB, whole redistributable (uncompressed, CUDA libraries included) <= 800 MB
  LIC   the project's LICENSE, NOTICE and THIRD_PARTY_NOTICES.md at the root and in share/doc/mossinfer/
  CRT   Windows: the MSVC runtime (vcruntime140, msvcp140 and every other one the binaries import) ships in bin/
  find_package: optionally builds and runs examples/consumer-cmake against the tree, with the CUDA toolkit removed
        from PATH / LD_LIBRARY_PATH so that only the package's own libraries can satisfy it

Usage:
  check_package.py --prefix <install dir> [--consumer <source dir>] [--consumer-build <dir>] [--run-consumer]
  check_package.py --archive <mossinfer-*.zip|.tar.gz> [...same options]
Only the standard library; Windows needs dumpbin on PATH (a VS developer prompt).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

MB = 1000 * 1000
ENGINE_LIMIT = 150 * MB
TOTAL_LIMIT = 800 * MB
FORBIDDEN = re.compile(r"torch|c10|python", re.IGNORECASE)
IS_WINDOWS = os.name == "nt"


# The C++ runtime DLLs of MSVC's /MD (the UCRT, api-ms-win-*, ships with Windows 10 and newer).
MSVC_RUNTIME = re.compile(r"^(vcruntime\d+(_\d+)?|msvcp\d+(_\w+)?|concrt\d+|vcomp\d+)\.dll$", re.IGNORECASE)
REQUIRED_MSVC_RUNTIME = ("vcruntime140.dll", "msvcp140.dll")
CUDA_DIR = re.compile(r"cuda|nvidia gpu computing toolkit", re.IGNORECASE)


def without_cuda_dirs(value):
    """A PATH-style list without the CUDA toolkit's directories."""
    return os.pathsep.join(p for p in (value or "").split(os.pathsep) if p and not CUDA_DIR.search(p))


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def is_elf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def is_pe(path):
    with open(path, "rb") as f:
        return f.read(2) == b"MZ"


def binaries(prefix):
    out = []
    for root, _, files in os.walk(prefix):
        for name in files:
            p = os.path.join(root, name)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            if (IS_WINDOWS and name.lower().endswith((".dll", ".exe")) and is_pe(p)) or (not IS_WINDOWS and is_elf(p)):
                out.append(p)
    return sorted(out)


def dependencies(path):
    if IS_WINDOWS:
        text = run(["dumpbin", "/nologo", "/dependents", path])
        return [l.strip() for l in text.splitlines() if l.strip().lower().endswith(".dll")]
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run(["readelf", "-d", path]))
    # ldd also shows what the loader resolves transitively (it may fail for libraries the host cannot load).
    try:
        resolved = [l.split()[0] for l in run(["ldd", path]).splitlines() if l.strip()]
    except subprocess.CalledProcessError:
        resolved = []
    return needed + resolved


def exported_symbols(lib):
    if IS_WINDOWS:
        text = run(["dumpbin", "/nologo", "/exports", lib])
        # Rows: ordinal hint RVA name
        return [m.group(1) for m in re.finditer(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)", text, re.MULTILINE)]
    text = run(["nm", "-D", "--defined-only", lib])
    return [l.split()[-1] for l in text.splitlines() if l.strip()]


def tree_size(prefix):
    total = 0
    for root, _, files in os.walk(prefix):
        for name in files:
            p = os.path.join(root, name)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def extract(archive, into):
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(into, filter="tar")
            except TypeError:  # Python without extraction filters (< 3.10.12)
                t.extractall(into)
    entries = os.listdir(into)
    return os.path.join(into, entries[0]) if len(entries) == 1 else into


def check_consumer(prefix, source, build_dir, run_it):
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    subprocess.run(["cmake", "-S", source, "-B", build_dir, *gen, "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_PREFIX_PATH={os.path.abspath(prefix)}"], check=True)
    subprocess.run(["cmake", "--build", build_dir, "--config", "Release"], check=True)
    if run_it:
        # Only the package may provide the libraries: no CUDA toolkit on PATH (Windows DLL search) or LD_LIBRARY_PATH,
        # and the package's bin/ is not added either (the consumer copies its DLLs next to itself on Windows; on Linux
        # RUNPATH finds them).
        env = dict(os.environ)
        env["PATH"] = without_cuda_dirs(env.get("PATH"))
        env["LD_LIBRARY_PATH"] = without_cuda_dirs(env.get("LD_LIBRARY_PATH"))
        env.pop("CUDA_PATH", None)
        ctest = shutil.which("ctest", path=env["PATH"]) or shutil.which("ctest")
        subprocess.run([ctest, "--test-dir", build_dir, "-C", "Release", "--output-on-failure", "-V"], check=True,
                       env=env)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prefix")
    src.add_argument("--archive")
    ap.add_argument("--consumer", help="examples/consumer-cmake source dir to build against the tree")
    ap.add_argument("--consumer-build", default=None)
    ap.add_argument("--run-consumer", action="store_true", help="also run it (needs no GPU)")
    args = ap.parse_args()

    tmp = None
    if args.archive:
        tmp = tempfile.mkdtemp(prefix="mi_pkg_")
        prefix = extract(args.archive, tmp)
        print(f"archive {os.path.basename(args.archive)}: {os.path.getsize(args.archive) / MB:.1f} MB compressed")
    else:
        prefix = args.prefix
    failures = []

    bins = binaries(prefix)
    if not bins:
        failures.append("no binaries found")
    for b in bins:
        deps = dependencies(b)
        bad = sorted({d for d in deps if FORBIDDEN.search(d)})
        rel = os.path.relpath(b, prefix)
        print(f"AC1 {rel}: {len(set(deps))} deps" + (f", FORBIDDEN {bad}" if bad else ", ok"))
        if bad:
            failures.append(f"AC1: {rel} depends on {bad}")

    lib_name = "mossinfer.dll" if IS_WINDOWS else None
    libs = [b for b in bins if os.path.basename(b) == lib_name] if IS_WINDOWS else [
        b for b in bins if re.match(r"libmossinfer\.so(\.\d+)*$", os.path.basename(b))]
    if not libs:
        failures.append("shared library libmossinfer.so / mossinfer.dll not in the tree")
    for lib in libs:
        syms = exported_symbols(lib)
        foreign = [s for s in syms if not s.startswith("mi_")]
        print(f"ABI {os.path.relpath(lib, prefix)}: {len(syms)} exported symbols, {len(foreign)} not mi_*")
        if not syms or foreign:
            failures.append(f"ABI: exports other than mi_*: {foreign[:20]}")
        size = os.path.getsize(lib)
        print(f"AC17 engine library {size / MB:.1f} MB (limit {ENGINE_LIMIT / MB:.0f} MB)")
        if size > ENGINE_LIMIT:
            failures.append(f"AC17: engine library {size / MB:.1f} MB > {ENGINE_LIMIT / MB:.0f} MB")

    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        for rel in (name, os.path.join("share", "doc", "mossinfer", name)):
            present = os.path.isfile(os.path.join(prefix, rel))
            print(f"LIC {rel}: {'ok' if present else 'MISSING'}")
            if not present:
                failures.append(f"LIC: {rel} missing (Apache-2.0 requires it in every distribution)")
    third_party = os.path.join("share", "doc", "mossinfer", "third_party")
    for rel in ("cutlass/LICENSE.txt", "flash-attention/LICENSE", "sageattention/LICENSE", "cuda/EULA.txt"):
        rel = os.path.join(third_party, *rel.split("/"))
        present = os.path.isfile(os.path.join(prefix, rel))
        print(f"LIC {rel}: {'ok' if present else 'MISSING'}")
        if not present:
            failures.append(f"LIC: {rel} missing (a bundled component's licence must ship with it)")

    if IS_WINDOWS:
        bin_dir = os.path.join(prefix, "bin")
        shipped = {n.lower() for n in os.listdir(bin_dir)} if os.path.isdir(bin_dir) else set()
        imported = set()
        for b in bins:
            imported |= {os.path.basename(d).lower() for d in dependencies(b) if MSVC_RUNTIME.match(os.path.basename(d))}
        for dll in sorted(imported | set(REQUIRED_MSVC_RUNTIME)):
            print(f"CRT bin/{dll}: {'ok' if dll in shipped else 'MISSING'}")
            if dll not in shipped:
                failures.append(f"CRT: bin/{dll} missing (InstallRequiredSystemLibraries)")

    total = tree_size(prefix)
    print(f"AC17 redistributable {total / MB:.1f} MB uncompressed (limit {TOTAL_LIMIT / MB:.0f} MB)")
    for b in bins:
        print(f"     {os.path.getsize(b) / MB:8.1f} MB  {os.path.relpath(b, prefix)}")
    if total > TOTAL_LIMIT:
        failures.append(f"AC17: redistributable {total / MB:.1f} MB > {TOTAL_LIMIT / MB:.0f} MB")

    if args.consumer:
        build_dir = args.consumer_build or os.path.join(tmp or tempfile.mkdtemp(prefix="mi_consumer_"), "_consumer")
        try:
            check_consumer(prefix, args.consumer, build_dir, args.run_consumer)
            print("find_package consumer: ok")
        except subprocess.CalledProcessError as e:
            failures.append(f"consumer build/run failed: {e}")

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
