#!/usr/bin/env python3
"""Install-tree test of the cuBLASLt search (src/backend/cuda_libs.h). Standard library only; needs no GPU.

    python3 tests/server/test_install_tree.py --build build-dev --work build-dev/install_tree_test \\
        [--check-package .github/scripts/check_package.py]

Installs the build twice: the full tree (every component, cuBLASLt bundled) and the system-CUDA variant
(`--component Unspecified`, no CUDA library). Then asks each tree's moss-server `hello` which cuBLASLt copy it would
use under MOSSINFER_CUDA_LIBS / MOSSINFER_CUBLASLT, and runs check_package.py on both trees (--system-cuda on the
second). The system expectations follow what this machine has: with no system cuBLASLt, `system` must fail cleanly.

The system-CUDA variant is a release-preset product (static cudart). With --shared-cudart (a dev build) the slim tree
also needs a system cudart, which the bundled one in the full tree otherwise provides; when the slim moss-server cannot
start for lack of it, its checks are skipped with a note instead of failing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

TIMEOUT = 60

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def install(cmake, build, prefix, component=None):
    shutil.rmtree(prefix, ignore_errors=True)
    cmd = [cmake, "--install", build, "--prefix", prefix] + (["--component", component] if component else [])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def hello(prefix, **env_vars):
    exe = os.path.join(prefix, "bin", "moss-server" + (".exe" if os.name == "nt" else ""))
    env = {k: v for k, v in os.environ.items() if k not in ("MOSSINFER_CUDA_LIBS", "MOSSINFER_CUBLASLT")}
    env.update(env_vars)
    p = subprocess.run([exe], input='{"id": 1, "method": "hello", "params": {"protocol_version": 1}}\n',
                       capture_output=True, text=True, env=env, timeout=TIMEOUT)
    for line in p.stdout.splitlines():
        msg = json.loads(line)
        if msg.get("id") == 1:
            if "result" not in msg:
                raise SystemExit(f"hello failed: {msg}")
            if p.returncode != 0:
                raise SystemExit(f"moss-server exited with {p.returncode}: {p.stderr}")
            return msg["result"]
    raise SystemExit(f"no hello answer (exit {p.returncode}): {p.stdout!r} {p.stderr!r}")


def inside(path, prefix):
    path, prefix = os.path.realpath(path), os.path.realpath(prefix)
    return os.path.commonpath([path, prefix]) == prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cmake", default="cmake")
    ap.add_argument("--check-package")
    ap.add_argument("--cuda-major", help="passed to check_package.py")
    ap.add_argument("--shared-cudart", action="store_true", help="a MOSSINFER_STATIC_CUDART=OFF build (dev preset)")
    args = ap.parse_args()
    full = os.path.join(args.work, "full")
    slim = os.path.join(args.work, "slim")
    install(args.cmake, args.build, full)
    install(args.cmake, args.build, slim, "Unspecified")

    # What this machine offers outside any package: the system-CUDA tree's own search.
    try:
        system = hello(slim, MOSSINFER_CUDA_LIBS="system")
    except SystemExit as e:
        if not args.shared_cudart:
            raise
        print(f"SKIP the system-CUDA tree of a shared-cudart build cannot start here (no system cudart): {e}")
        system = None
    slim_runs = system is not None
    if system is None:
        system = hello(full, MOSSINFER_CUDA_LIBS="system")
    lt = system["probe"]["cublaslt"]
    have_system = bool(lt["source"])
    print(f"system cuBLASLt: {lt['path'] or lt['error']}", flush=True)
    check(system["capabilities"]["system_cuda"] is True, "capabilities.system_cuda")
    if have_system:
        check(lt["source"] == "system" and os.path.isfile(lt["path"]) and not inside(lt["path"], args.work),
              f"system: a system copy outside the trees ({lt})")
    else:
        check(lt["version"] == 0 and "(system)" in lt["error"], f"system: a clean error ({lt})")

    # Full tree, bundled: the copy in the tree's lib/ (bin/ on Windows).
    lt = hello(full, MOSSINFER_CUDA_LIBS="bundled")["probe"]["cublaslt"]
    check(lt["source"] == "bundled" and inside(lt["path"], full) and lt["version"] > 0 and lt["error"] == "",
          f"full, bundled: the tree's copy ({lt})")
    bundled_file = lt["path"]

    # Full tree, system: never the bundled copy, even through the library's RUNPATH (a shared-cudart build keeps
    # $ORIGIN there) or an executable's.
    lt = hello(full, MOSSINFER_CUDA_LIBS="system")["probe"]["cublaslt"]
    if have_system:
        check(lt["source"] == "system" and not inside(lt["path"], full), f"full, system: not the tree's copy ({lt})")
    else:
        check(lt["source"] == "" and "(system)" in lt["error"], f"full, system: a clean error ({lt})")

    # Full tree, auto (default): always its own copy, even where a system copy exists.
    lt = hello(full)["probe"]["cublaslt"]
    check(lt["source"] == "bundled" and os.path.realpath(lt["path"]) == os.path.realpath(bundled_file), f"full, auto: {lt}")

    # MOSSINFER_CUBLASLT: exactly that file.
    lt = hello(full, MOSSINFER_CUBLASLT=bundled_file, MOSSINFER_CUDA_LIBS="system")["probe"]["cublaslt"]
    check(lt["source"] == "env" and os.path.realpath(lt["path"]) == os.path.realpath(bundled_file), f"full, env: {lt}")
    lt = hello(full, MOSSINFER_CUBLASLT=os.path.join(args.work, "missing", "libcublasLt.so.1"))["probe"]["cublaslt"]
    check(lt["source"] == "" and "missing (1): " in lt["error"] and lt["error"].endswith("(env)"),
          f"full, env to a missing file: a clean error ({lt})")
    lt = hello(full, MOSSINFER_CUBLASLT="libcublasLt.so.13")["probe"]["cublaslt"]
    check(lt["source"] == "" and "not an absolute path" in lt["error"], f"full, relative env: refused ({lt})")

    # System-CUDA tree: auto falls through to the system copy (it has no bundled one); bundled says so (no crash).
    if slim_runs:
        lt = hello(slim)["probe"]["cublaslt"]
        check(lt["source"] == ("system" if have_system else ""), f"slim, auto: {lt}")
        lt = hello(slim, MOSSINFER_CUDA_LIBS="bundled")["probe"]["cublaslt"]
        check(lt["source"] == "" and lt["version"] == 0 and "missing (" in lt["error"] and "(bundled)" in lt["error"],
              f"slim, bundled: a clean error ({lt})")
        lt = hello(slim, MOSSINFER_CUDA_LIBS="sometimes")["probe"]["cublaslt"]
        check(lt["source"] == "" and "MOSSINFER_CUDA_LIBS=sometimes" in lt["error"], f"slim, bad policy: {lt}")

    if args.check_package:
        major = ["--cuda-major", args.cuda_major] if args.cuda_major else []
        for prefix, extra in ((full, major), (slim, ["--system-cuda"])):
            p = subprocess.run([sys.executable, args.check_package, "--prefix", prefix] + extra, capture_output=True,
                               text=True)
            sys.stdout.write(p.stdout)
            check(p.returncode == 0, f"check_package.py {' '.join(extra)} {prefix}: {p.stderr.strip()}")

    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
