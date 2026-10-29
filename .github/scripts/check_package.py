#!/usr/bin/env python3
"""Checks an installed MossInfer tree or a CPack archive (CI and local).

  AC1   no torch / c10 / python among the dependencies of any shipped binary (readelf+ldd / dumpbin /dependents)
  ABI   the shared library exports exactly the C API functions of include/mossinfer/mossinfer.h (its MOSSINFER_API
        declarations), plus mi_internal_cli_main, moss-infer's entry point (nm -D / dumpbin /exports)
  AC17  engine library <= 150 MB, whole redistributable (uncompressed, CUDA libraries included) <= 800 MB;
        900 MB for CUDA 12 packages, whose bundled cuBLASLt alone is ~752 MB (CUDA 12.8)
  LIC   the project's LICENSE, NOTICE and THIRD_PARTY_NOTICES.md at the root and in share/doc/mossinfer/
  CRT   Windows: the MSVC runtime (vcruntime140, msvcp140 and every other one the binaries import) ships in bin/
  LT    cuBLASLt is loaded at run time through the engine's shim (src/backend/cublaslt_shim.cpp), never linked: no
        shipped binary has a DT_NEEDED / import-table entry for it or imports a cublasLt* symbol (readelf -d, nm -D
        --undefined-only / dumpbin /dependents, /imports). The full package bundles it (with the CUDA EULA); the
        --system-cuda variant ships no CUDA library at all and needs no EULA
  find_package: optionally builds and runs examples/consumer-cmake against the tree, with the CUDA toolkit removed
        from PATH / LD_LIBRARY_PATH so that only the package's own libraries can satisfy it

Usage:
  check_package.py --prefix <install dir> [--system-cuda] [--consumer <source dir>] [--consumer-build <dir>]
                   [--run-consumer]
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
# CUDA 12's cuBLASLt (12.8) alone is ~752 MB uncompressed; we cannot shrink NVIDIA's library (user decision 2026-10-22).
TOTAL_LIMIT_CUDA12 = 900 * MB
CUBLASLT_12 = re.compile(r"^(libcublasLt\.so\.12|cublasLt64_12\.dll)", re.IGNORECASE)
FORBIDDEN = re.compile(r"torch|c10|python", re.IGNORECASE)
CUBLASLT_FILE = re.compile(r"^(libcublasLt\.so|cublasLt64_)", re.IGNORECASE)
# The CUDA toolkit libraries a package may bundle (the system-CUDA variant ships none of them).
CUDA_LIB_FILE = re.compile(r"^(lib)?(cublas|cudart)", re.IGNORECASE)
CUBLASLT_DEP = re.compile(r"cublasLt", re.IGNORECASE)
IS_WINDOWS = os.name == "nt"
# ABI: the shared library exports exactly the C API (every MOSSINFER_API declaration of the installed header) plus
# moss-infer's internal entry point.
HEADER = "include/mossinfer/mossinfer.h"
API_DECL = re.compile(r"^\s*MOSSINFER_API\b[^;(]*?\b(\w+)\s*\(", re.MULTILINE)
INTERNAL_CLI_ENTRY = "mi_internal_cli_main"
# The component API's frozen functions (2.1): the header must declare every one, so the export check covers them.
COMPONENT_API = {
    "mi_session_params_init", "mi_session_open", "mi_session_close",
    "mi_text_params_init", "mi_encode_text",
    "mi_conditioning_create", "mi_conditioning_query", "mi_conditioning_prompt_tokens", "mi_conditioning_tensor",
    "mi_conditioning_free",
    "mi_run_params_init", "mi_run_begin", "mi_run_query", "mi_run_sigmas", "mi_run_adopt_staging", "mi_run_staging",
    "mi_step_params_init", "mi_denoise_step", "mi_run_end",
    "mi_vae_encode", "mi_vae_decode",
    "mi_schedule", "mi_sampler_prediction", "mi_initial_noise", "mi_image_resize",
}
# The check's functions (2.1, MI_CAP_CHECK), held to the same rule.
CHECK_API = {"mi_check", "mi_check_report_info", "mi_check_report_text", "mi_check_report_free"}


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


def direct_dependencies(path):
    """The binary's own DT_NEEDED / import-table entries (not what they load in turn)."""
    if IS_WINDOWS:
        return dependencies(path)
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run(["readelf", "-d", path]))


def imported_symbols(path):
    if IS_WINDOWS:
        # Rows under each DLL: "  hint  name" (or "Ordinal n"); the DLL lines end in .dll.
        text = run(["dumpbin", "/nologo", "/imports", path])
        return [m.group(1) for m in re.finditer(r"^\s+[0-9A-F]+\s+(\w+)\s*$", text, re.MULTILINE)]
    text = run(["nm", "-D", "--undefined-only", path])
    return [l.split()[-1].split("@")[0] for l in text.splitlines() if l.strip()]


def exported_symbols(lib):
    if IS_WINDOWS:
        text = run(["dumpbin", "/nologo", "/exports", lib])
        # Rows: ordinal hint RVA name
        return [m.group(1) for m in re.finditer(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)", text, re.MULTILINE)]
    text = run(["nm", "-D", "--defined-only", lib])
    return [l.split()[-1] for l in text.splitlines() if l.strip()]


def c_api_functions(prefix):
    """The names of the functions the installed header declares MOSSINFER_API."""
    path = os.path.join(prefix, *HEADER.split("/"))
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(API_DECL.findall(f.read()))


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
    ap.add_argument("--system-cuda", action="store_true",
                    help="the system-CUDA variant: no CUDA library in the tree, so no CUDA EULA either")
    ap.add_argument("--cuda-major", help="the flavour's CUDA major (default: from the -cu<N>- in the package name)")
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
    api = c_api_functions(prefix)
    if not api:
        failures.append(f"ABI: no MOSSINFER_API functions found in {HEADER}")
    elif COMPONENT_API - api:
        failures.append(f"ABI: component API functions not declared in {HEADER}: {sorted(COMPONENT_API - api)}")
    elif CHECK_API - api:
        failures.append(f"ABI: check functions not declared in {HEADER}: {sorted(CHECK_API - api)}")
    for lib in libs:
        syms = set(exported_symbols(lib))
        # mi_internal_cli_main: moss-infer's entry when it links the library (MOSSINFER_CLI_SHARED); not C API.
        # cudart_static.lib (Windows) exports NvOptimusEnablementCuda itself, a DWORD hint for Optimus laptops.
        foreign = sorted(syms - api - {INTERNAL_CLI_ENTRY, "NvOptimusEnablementCuda"})
        missing = sorted(api - syms)
        print(f"ABI {os.path.relpath(lib, prefix)}: {len(syms)} exported symbols ({len(api)} C API functions in "
              f"{HEADER}" + (f", {INTERNAL_CLI_ENTRY}" if INTERNAL_CLI_ENTRY in syms else "") + f"), "
              f"{len(foreign)} foreign, {len(missing)} missing")
        if foreign:
            failures.append(f"ABI: exports beyond the C API and {INTERNAL_CLI_ENTRY}: {foreign[:20]}")
        if missing:
            failures.append(f"ABI: C API functions not exported: {missing[:20]}")
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
    for rel in ("cutlass/LICENSE.txt", "flash-attention/LICENSE", "sageattention/LICENSE") + \
            (() if args.system_cuda else ("cuda/EULA.txt",)):
        rel = os.path.join(third_party, *rel.split("/"))
        present = os.path.isfile(os.path.join(prefix, rel))
        print(f"LIC {rel}: {'ok' if present else 'MISSING'}")
        if not present:
            failures.append(f"LIC: {rel} missing (a bundled component's licence must ship with it)")

    # LT: every cuBLASLt call goes through the engine's shim; the library itself only in the full package.
    files = [os.path.join(r, n) for r, _, names in os.walk(prefix) for n in names]
    cuda_files = sorted(os.path.relpath(f, prefix) for f in files if CUDA_LIB_FILE.match(os.path.basename(f)))
    cublaslt_files = [f for f in cuda_files if CUBLASLT_FILE.match(os.path.basename(f))]
    eula = os.path.join("share", "doc", "mossinfer", "third_party", "cuda", "EULA.txt")
    if args.system_cuda:
        print(f"LT system-CUDA variant: {len(cuda_files)} CUDA library files" + (f" {cuda_files}" if cuda_files else ""))
        if cuda_files:
            failures.append(f"LT: the system-CUDA variant ships CUDA libraries {cuda_files}")
        if os.path.exists(os.path.join(prefix, eula)):
            failures.append(f"LT: the system-CUDA variant must not ship {eula} (it redistributes no CUDA library)")
    else:
        # Exactly the file the loader opens: lib/libcublasLt.so.<major> (Windows: bin/cublasLt64_<major>.dll), with
        # the flavour's major when the package name tells it (…-cu13-…).
        major = args.cuda_major or (re.search(r"-cu(\d+)-", os.path.basename(os.path.normpath(args.archive or prefix)))
                                    or [None, r"\d+"])[1]
        want = re.compile(rf"^bin[\\/]cublasLt64_{major}\.dll$" if IS_WINDOWS else rf"^lib/libcublasLt\.so\.{major}$",
                          re.IGNORECASE)
        ok = len(cublaslt_files) == 1 and want.match(cublaslt_files[0])
        print(f"LT bundled cuBLASLt: {cublaslt_files or 'MISSING'}" + ("" if ok else f" (want exactly {want.pattern})"))
        if not ok:
            failures.append(f"LT: the full package must bundle exactly {want.pattern}, has {cublaslt_files}")
    for b in bins:
        if CUDA_LIB_FILE.match(os.path.basename(b)):
            continue  # NVIDIA's own libraries
        rel = os.path.relpath(b, prefix)
        needed = [d for d in direct_dependencies(b) if CUBLASLT_DEP.search(os.path.basename(d))]
        imported = sorted({sym for sym in imported_symbols(b) if sym.startswith("cublasLt")})
        print(f"LT {rel}: " + ("ok" if not needed and not imported else f"links cuBLASLt {needed} {imported[:10]}"))
        if needed:
            failures.append(f"LT: {rel} has a load-time dependency on {needed} (cuBLASLt must go through the shim)")
        if imported:
            failures.append(f"LT: {rel} imports {imported[:10]} directly (add them to src/backend/cublaslt_shim.cpp)")

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
    cuda12 = any(CUBLASLT_12.match(n) for _, _, files in os.walk(prefix) for n in files)
    total_limit = TOTAL_LIMIT_CUDA12 if cuda12 else TOTAL_LIMIT
    print(f"AC17 redistributable {total / MB:.1f} MB uncompressed (limit {total_limit / MB:.0f} MB"
          f"{', CUDA 12 cuBLASLt bundled' if cuda12 else ''})")
    for b in bins:
        print(f"     {os.path.getsize(b) / MB:8.1f} MB  {os.path.relpath(b, prefix)}")
    if total > total_limit:
        failures.append(f"AC17: redistributable {total / MB:.1f} MB > {total_limit / MB:.0f} MB")

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
