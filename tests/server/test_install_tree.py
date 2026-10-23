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


def install(cmake, build, prefix, component=None, certs=None):
    """certs: the certificate source directory (packaging.cmake's MOSSINFER_QUALITY_CERTS_SOURCE); None: the build's."""
    shutil.rmtree(prefix, ignore_errors=True)
    cmd = [cmake, "--install", build, "--prefix", prefix] + (["--component", component] if component else [])
    env = {k: v for k, v in os.environ.items() if k != "MOSSINFER_QUALITY_CERTS_SOURCE"}
    if certs is not None:
        env["MOSSINFER_QUALITY_CERTS_SOURCE"] = certs
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=env)


CERT_DIR = os.path.join("share", "mossinfer", "quality_certs")
FAKE_CERT = "test-only.mossinfer-cert.json"


def fake_cert_source(work):
    """A certificate source directory with one syntactically valid (and matching nothing) certificate, plus a file
    that must not be installed."""
    src = os.path.join(work, "cert_source")
    shutil.rmtree(src, ignore_errors=True)
    os.makedirs(src)
    with open(os.path.join(src, FAKE_CERT), "w", encoding="utf-8") as f:
        json.dump({"schema": "mossinfer.quality_cert/test", "note": "install-tree test only; certifies nothing"}, f)
    with open(os.path.join(src, "README.md"), "w", encoding="utf-8") as f:
        f.write("not a certificate\n")
    if os.name == "posix":  # a symbolic link is skipped (packaging.cmake)
        os.symlink(FAKE_CERT, os.path.join(src, "link.mossinfer-cert.json"))
    return src


def source_certs(build):
    """The *.mossinfer-cert.json files of the build's certificate directory (packaging.cmake), possibly none."""
    src = None
    with open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("MOSSINFER_QUALITY_CERTS_DIR:"):
                src = line.split("=", 1)[1].strip()
    if not src or not os.path.isdir(src):
        return []
    return sorted(n for n in os.listdir(src) if n.endswith(".mossinfer-cert.json"))


def installed_certs(prefix):
    d = os.path.join(prefix, CERT_DIR)
    return sorted(os.listdir(d)) if os.path.isdir(d) else None


def check_package_run(script, prefix, extra):
    p = subprocess.run([sys.executable, script, "--prefix", prefix] + extra, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def check_certificates(args, full, slim):
    """Packaging of share/mossinfer/quality_certs: the full tree got the fake source's certificate only, the slim
    tree the build's own (none yet: no directory at all), and check_package.py lists them and refuses strangers."""
    got = installed_certs(full)
    check(got == [FAKE_CERT], f"full: {CERT_DIR} holds exactly the source's certificate ({got})")
    own = source_certs(args.build)
    got = installed_certs(slim)
    check(got == (own or None), f"slim: {CERT_DIR} holds the build's certificates {own} ({got})")
    if not args.check_package:
        return
    major = ["--cuda-major", args.cuda_major] if args.cuda_major else []
    rc, out, err = check_package_run(args.check_package, full, major)
    rel = os.path.join(CERT_DIR, FAKE_CERT)
    check(rc == 0 and f"CERT {rel}: ok" in out and f"CERT {CERT_DIR}: 1 certificate(s)" in out,
          f"check_package.py lists the certificate: {err.strip()}")
    cert_dir = os.path.join(full, CERT_DIR)
    for name, body, want in (("notes.txt", "x", "is not a .mossinfer-cert.json file"),
                             ("broken.mossinfer-cert.json", "{not json", "is not valid JSON")):
        path = os.path.join(cert_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        try:
            rc, out, err = check_package_run(args.check_package, full, major)
        finally:
            os.remove(path)
        check(rc != 0 and want in err, f"check_package.py refuses {name} ({rc}: {err.strip()[-200:]})")
    if os.name == "posix":
        link = os.path.join(cert_dir, "link.mossinfer-cert.json")
        os.symlink(FAKE_CERT, link)
        try:
            rc, out, err = check_package_run(args.check_package, full, major)
        finally:
            os.remove(link)
        check(rc != 0 and "is a symbolic link" in err, f"check_package.py refuses a symbolic link ({rc}: {err.strip()[-200:]})")


def gpu_certificate_search(full, weights):
    """GPU (MOSSINFER_GPU_TESTS=1): the installed engine searches the tree's certificate directory. A 256^2, 2-step
    generation on the High tier (which needs a certificate) names the fake certificate in its quality reason; the fake
    one certifies nothing."""
    base = os.path.join(weights, "base")
    if not os.path.isdir(os.path.join(base, "transformer")):
        print(f"SKIP GPU certificate search: no base bundle under {weights}")
        return
    exe = os.path.abspath(os.path.join(full, "bin", "moss-server" + (".exe" if os.name == "nt" else "")))
    out_png = os.path.join(os.path.dirname(os.path.abspath(full)), "cert_search.png")
    p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         env=clean_environment(), cwd=os.path.dirname(os.path.abspath(full)))
    try:
        for req in ({"id": 1, "method": "load", "params": {"model": base, "tier": "high"}},
                    {"id": 2, "method": "generate", "params": {"prompt": "a red apple", "width": 256, "height": 256,
                                                               "steps": 2, "output": out_png}}):
            p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            msg = json.loads(line)
            if msg.get("id") in (1, 2) and "error" in msg:
                raise SystemExit(f"GPU certificate search: {msg}")
            if msg.get("event") == "done":
                done = msg
                break
        p.stdin.write('{"id": 3, "method": "shutdown"}\n')
        p.stdin.flush()
        p.communicate(timeout=TIMEOUT)
    except Exception:
        p.kill()
        p.wait()
        raise
    check(done is not None and done.get("status") == "ok", f"GPU: the installed server generated ({done and done.get('status')})")
    if done is None or done.get("status") != "ok":
        return
    reason = done["report"]["quality"].get("reason", "")
    print(f"GPU quality reason: {reason}", flush=True)
    check(FAKE_CERT in reason and "install-tree certificates not searched" not in reason,
          "GPU: the install tree's certificate directory was searched (the reason names the fake certificate)")
    check(done.get("certified") is False, "GPU: the fake certificate certifies nothing")


def clean_environment():
    """The test's environment without anything that could point the installed binaries back at the build tree. An
    LD_LIBRARY_PATH with an empty entry (CI's "/usr/local/cuda-13.3/lib64:") searches the working directory, which
    under ctest is the build tree: the installed moss-server would load the build tree's libmossinfer."""
    drop = ("MOSSINFER_CUDA_LIBS", "MOSSINFER_CUBLASLT", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")
    return {k: v for k, v in os.environ.items() if k not in drop}


def loaded_engine(pid):
    """Linux: the libmossinfer file the process mapped (None elsewhere or when unreadable)."""
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                parts = line.split(maxsplit=5)  # address perms offset dev inode [path]
                path = parts[5].strip() if len(parts) == 6 else ""
                if os.path.basename(path).startswith("libmossinfer.so"):
                    return path
    except OSError:
        pass
    return None


def hello(prefix, **env_vars):
    exe = os.path.abspath(os.path.join(prefix, "bin", "moss-server" + (".exe" if os.name == "nt" else "")))
    env = clean_environment()
    env.update(env_vars)
    p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, cwd=os.path.dirname(os.path.abspath(prefix)))  # never the build tree
    try:
        p.stdin.write('{"id": 1, "method": "hello", "params": {"protocol_version": 1}}\n')
        p.stdin.flush()
        answer = None
        for line in p.stdout:
            msg = json.loads(line)
            if msg.get("id") == 1:
                answer = msg
                break
        engine = loaded_engine(p.pid)  # while the process is alive
        # communicate() closes stdin (the server then exits); closing it here first trips Python 3.13
        out, err = p.communicate(timeout=TIMEOUT)
    except Exception:
        p.kill()
        p.wait()
        raise
    if answer is None:
        raise SystemExit(f"no hello answer (exit {p.returncode}): {out!r} {err!r}")
    if "result" not in answer:
        raise SystemExit(f"hello failed: {answer}")
    if p.returncode != 0:
        raise SystemExit(f"moss-server exited with {p.returncode}: {err}")
    if os.name == "posix" and sys.platform.startswith("linux"):
        # The installed binary must run the installed engine, never the build tree's.
        if engine is None or not inside(engine, prefix):
            raise SystemExit(f"{exe} loaded the engine {engine}, not the one in {prefix}")
    return answer["result"]


def cli_help(prefix):
    """The tree's moss-infer starts and prints its usage. Linux: when it links libmossinfer (MOSSINFER_CLI_SHARED, the
    release presets), it resolves the tree's copy, never the build tree's."""
    exe = os.path.abspath(os.path.join(prefix, "bin", "moss-infer" + (".exe" if os.name == "nt" else "")))
    if not os.path.isfile(exe):
        return
    env = clean_environment()
    cwd = os.path.dirname(os.path.abspath(prefix))  # never the build tree
    p = subprocess.run([exe, "--help"], capture_output=True, text=True, env=env, cwd=cwd, timeout=TIMEOUT)
    check(p.returncode == 0 and "usage: moss-infer" in p.stdout, f"{prefix}: moss-infer --help ({p.returncode} {p.stderr!r})")
    if sys.platform.startswith("linux"):
        ldd = subprocess.run(["ldd", exe], capture_output=True, text=True, env=env, cwd=cwd).stdout
        engine = [l.split("=>")[1].split()[0] for l in ldd.splitlines() if "libmossinfer" in l and "=>" in l]
        if engine:
            check(inside(engine[0], prefix), f"{prefix}: moss-infer links the tree's libmossinfer ({engine[0]})")


def inside(path, prefix):
    path, prefix = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(prefix))
    try:
        return os.path.commonpath([path, prefix]) == prefix
    except ValueError:  # Windows: different drives
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cmake", default="cmake")
    ap.add_argument("--check-package")
    ap.add_argument("--cuda-major", help="passed to check_package.py")
    ap.add_argument("--shared-cudart", action="store_true", help="a MOSSINFER_STATIC_CUDART=OFF build (dev preset)")
    args = ap.parse_args()
    args.work = os.path.abspath(args.work)
    full = os.path.join(args.work, "full")
    slim = os.path.join(args.work, "slim")
    # The full tree with a fake certificate source, the slim one with the build's own (no directory yet: the install
    # must not fail on it).
    install(args.cmake, args.build, full, certs=fake_cert_source(args.work))
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

    cli_help(full)
    if slim_runs:
        cli_help(slim)

    if args.check_package:
        major =["--cuda-major", args.cuda_major] if args.cuda_major else []
        for prefix, extra in ((full, major), (slim, ["--system-cuda"])):
            p = subprocess.run([sys.executable, args.check_package, "--prefix", prefix] + extra, capture_output=True,
                               text=True)
            sys.stdout.write(p.stdout)
            check(p.returncode == 0, f"check_package.py {' '.join(extra)} {prefix}: {p.stderr.strip()}")

    check_certificates(args, full, slim)
    # The certificates are in the default component: the system-CUDA variant carries them too.
    install(args.cmake, args.build, slim, "Unspecified", certs=fake_cert_source(args.work))
    got = installed_certs(slim)
    check(got == [FAKE_CERT], f"slim with the fake source: {CERT_DIR} holds the certificate ({got})")
    if os.environ.get("MOSSINFER_GPU_TESTS", "0") == "1":
        gpu_certificate_search(full, os.environ.get("MOSS_WEIGHTS") or "/workspace/Models/MOSS-Image")

    if failures:
        print(f"{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
