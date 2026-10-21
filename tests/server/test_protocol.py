#!/usr/bin/env python3
"""moss-server protocol tests (docs/server-protocol.md). Standard library only.

    MOSS_SERVER=build/moss-server python3 tests/server/test_protocol.py
    MOSSINFER_GPU_TESTS=1 MOSS_SERVER=... python3 tests/server/test_protocol.py   # + generate / cancel on the GPU

The CPU tests never create an engine on a real bundle, so they run on machines without a GPU or weights. The GPU tests
need the base bundle under $MOSS_WEIGHTS (default /workspace/Models/MOSS-Image) and should run under the GPU lock.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

SERVER = os.environ.get("MOSS_SERVER", "")
GPU = os.environ.get("MOSSINFER_GPU_TESTS", "0") == "1"
WEIGHTS = os.environ.get("MOSS_WEIGHTS") or "/workspace/Models/MOSS-Image"
TIMEOUT = 30.0
GPU_TIMEOUT = 300.0


POSIX = os.name == "posix"


def process_gone(pid):
    """True once `pid` has exited (a zombie counts as exited)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


def wait_gone(pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_gone(pid):
            return True
        time.sleep(0.05)
    return False


class Server:
    """One moss-server process; stdout lines are parsed (and checked to be valid UTF-8 JSON) on a reader thread.

    `stderr`: None captures it in a temporary file; anything else is passed to Popen (DEVNULL, STDOUT). `preexec_fn`
    runs in the child before exec (POSIX), e.g. to close fd 2."""

    def __init__(self, args=(), stderr=None, preexec_fn=None):
        self.stderr = tempfile.TemporaryFile() if stderr is None else None
        self.proc = subprocess.Popen([SERVER, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr if stderr is None else stderr, preexec_fn=preexec_fn)
        self.lines = queue.Queue()
        self.bad_lines = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.next_id = 0

    def _read(self):
        for raw in self.proc.stdout:
            try:
                self.lines.put(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.bad_lines.append(raw)
        self.lines.put(None)  # end of output

    def send_raw(self, data: bytes):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def send(self, method, params=None, id=None, **extra):
        if id is None:
            self.next_id += 1
            id = self.next_id
        msg = {"id": id, "method": method}
        if params is not None:
            msg["params"] = params
        msg.update(extra)
        self.send_raw((json.dumps(msg) + "\n").encode("utf-8"))
        return id

    def read(self, timeout=TIMEOUT):
        try:
            msg = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise AssertionError("no output from moss-server within %.0f s" % timeout)
        if msg is None:
            raise AssertionError("moss-server closed its output; stderr:\n" + self.stderr_text())
        return msg

    def response(self, id, timeout=TIMEOUT, events=None):
        """The response to request `id`; events seen meanwhile go to `events`."""
        deadline = time.monotonic() + timeout
        while True:
            msg = self.read(max(0.1, deadline - time.monotonic()))
            if "event" in msg:
                if events is not None:
                    events.append(msg)
                continue
            if msg.get("id") == id:
                return msg
            raise AssertionError("unexpected message %r while waiting for id %r" % (msg, id))

    def call(self, method, params=None, **kw):
        return self.response(self.send(method, params, **kw))

    def stderr_text(self):
        if self.stderr is None:
            return "(stderr not captured)"
        self.stderr.seek(0)
        return self.stderr.read().decode("utf-8", "replace")

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.reader.join(timeout=5)
        for f in (self.proc.stdin, self.proc.stdout, self.stderr):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass


class ProtocolTestBase(unittest.TestCase):
    def setUp(self):
        if not SERVER or not os.path.isfile(SERVER):
            self.fail("set MOSS_SERVER to the moss-server binary (got %r)" % SERVER)
        self.server = Server()
        self.tmp = tempfile.mkdtemp(prefix="moss_server_test_")

    def tearDown(self):
        self.server.close()
        self.assertEqual(self.server.bad_lines, [], "stdout carried non-protocol lines")
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertError(self, msg, code, id=None):
        self.assertIn("error", msg, msg)
        self.assertNotIn("result", msg)
        self.assertEqual(msg["error"]["code"], code, msg)
        self.assertIsInstance(msg["error"]["message"], str)
        self.assertTrue(msg["error"]["message"])
        self.assertEqual(msg["id"], id, msg)

    def fake_bundle(self):
        path = os.path.join(self.tmp, "bundle")
        for c in ("text_encoder", "tokenizer", "transformer", "vae"):
            os.makedirs(os.path.join(path, c))
        return path


class CpuProtocolTest(ProtocolTestBase):
    def test_hello(self):
        r = self.server.call("hello", {"client": "test_protocol.py", "protocol_version": 1})
        res = r["result"]
        self.assertEqual(res["protocol_version"], 1)
        self.assertEqual(res["server"], "moss-server")
        self.assertRegex(res["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(res["abi_version"], 1)
        self.assertEqual(res["capabilities"], {"preview": False, "comfyui_models": False})
        probe = res["probe"]
        self.assertGreaterEqual(probe["device_count"], probe["supported_device_count"])
        self.assertIn(probe["recommended_flavour"], ("cu12", "cu13", ""))
        self.assertIn(probe["build_flavour"], ("cu12", "cu13"))
        self.assertTrue(probe["message"])
        self.assertNotIn("devices", probe)
        self.assertError(self.server.call("hello", {"protocol_version": 2}, id="v2"), "unsupported_protocol", "v2")

    def test_status_and_idle_commands_without_engine(self):
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})
        self.assertEqual(self.server.call("cancel")["result"], {"cancelling": False})
        self.assertEqual(self.server.call("unload")["result"], {"unloaded": False})
        self.assertError(self.server.call("generate", {"prompt": "a cat", "output": "x.png"}, id=7), "no_engine", 7)

    def test_malformed_input_never_kills_the_server(self):
        s = self.server
        s.send_raw(b"{not json\n")
        self.assertError(s.read(), "parse_error")
        s.send_raw(b"[1, 2]\n")
        self.assertError(s.read(), "invalid_request")
        s.send_raw(b'{"id": 3}\n')
        self.assertError(s.read(), "invalid_request", 3)
        s.send_raw(b'{"id": 1.5, "method": "hello"}\n')
        self.assertError(s.read(), "invalid_request")
        s.send_raw(b'{"id": "x", "method": "hello", "extra": 1}\n')
        self.assertError(s.read(), "invalid_request", "x")
        s.send_raw(b'{"id": 4, "method": "hello", "params": [1]}\n')
        self.assertError(s.read(), "invalid_params", 4)
        s.send_raw(b"\xff\xfe garbage \x00\n")
        self.assertError(s.read(), "parse_error")
        s.send_raw(b"   \n\n")  # blank lines are ignored
        s.send_raw(b'{"id": "crlf", "method": "status"}\r\n')
        self.assertEqual(s.read()["id"], "crlf")
        s.send_raw(b'{"id": 5, "method": "hello", "params": {"client": "\xc3\x28"}}\n')  # ill-formed UTF-8
        self.assertError(s.read(), "parse_error")
        s.send_raw(b'{"id": 5, "method": "hello", "params": {"client": "\\ud800"}}\n')  # unpaired surrogate escape
        self.assertError(s.read(), "parse_error")
        s.send_raw('{"id": "\u20ac", "method": "status"}\n'.encode("utf-8"))
        self.assertEqual(s.read(), {"id": "\u20ac", "result": {"state": "unloaded"}})
        s.send_raw(b"[" * 100000 + b"\n")
        self.assertError(s.read(), "parse_error")
        s.send_raw(b'{"id": 6, "method": "hello", "params": {"client": "' + b"a" * (2 << 20) + b'"}}\n')
        self.assertError(s.read(), "invalid_request")
        self.assertEqual(s.call("hello")["result"]["protocol_version"], 1)

    def test_unknown_method_and_params(self):
        self.assertError(self.server.call("frobnicate", id="m"), "method_not_found", "m")
        self.assertError(self.server.call("status", {"verbose": True}, id=9), "invalid_params", 9)
        self.assertError(self.server.call("load", {}, id=10), "invalid_params", 10)
        self.assertError(self.server.call("load", {"model": 5}, id=11), "invalid_params", 11)
        self.assertError(self.server.call("load", {"model": "/x", "mode": "fast"}, id=12), "invalid_params", 12)
        self.assertError(self.server.call("load", {"model": "/x", "device": -1}, id=13), "invalid_params", 13)

    def test_load_errors_are_engine_statuses(self):
        missing = os.path.join(self.tmp, "missing")
        r = self.server.call("load", {"model": missing}, id="a")
        self.assertError(r, "invalid_argument", "a")
        self.assertIn("not found", r["error"]["message"])
        partial = os.path.join(self.tmp, "partial")
        os.makedirs(os.path.join(partial, "transformer"))
        self.assertError(self.server.call("load", {"model": partial}, id="b"), "model", "b")
        ckpt = os.path.join(self.tmp, "moss.safetensors")
        open(ckpt, "wb").close()
        self.assertError(self.server.call("load", {"model": ckpt}, id="c"), "not_supported", "c")
        # Checked before CUDA is touched: a fake bundle with an invalid budget.
        r = self.server.call("load", {"model": self.fake_bundle(), "vram_budget_bytes": 1000}, id="d")
        self.assertError(r, "invalid_argument", "d")
        self.assertIn("vram_budget_bytes", r["error"]["message"])
        self.assertEqual(self.server.call("status")["result"]["state"], "unloaded")

    def test_shutdown(self):
        r = self.server.call("shutdown", id="bye")
        self.assertEqual(r, {"id": "bye", "result": {}})
        self.assertEqual(self.server.proc.wait(timeout=TIMEOUT), 0)

    def test_end_of_input_exits_cleanly(self):
        self.server.proc.stdin.close()
        self.assertEqual(self.server.proc.wait(timeout=TIMEOUT), 0)

    def test_version_flag(self):
        out = subprocess.run([SERVER, "--version"], capture_output=True, timeout=TIMEOUT)
        self.assertEqual(out.returncode, 0)
        self.assertRegex(out.stdout.decode(), r"^moss-server \d+\.\d+\.\d+ \(protocol 1, ABI 1\)")
        self.assertEqual(subprocess.run([SERVER, "--bogus"], capture_output=True, timeout=TIMEOUT).returncode, 2)
        for bad in (["--parent-pid"], ["--parent-pid", "abc"], ["--parent-pid", "0"], ["--parent-pid", "-5"],
                    ["--parent-pid", "99999999999"]):
            self.assertEqual(subprocess.run([SERVER, *bad], capture_output=True, timeout=TIMEOUT).returncode, 2, bad)

    def test_strings_with_nul_are_rejected(self):
        # The C API takes NUL-terminated strings: "/x\u0000y" must not become "/x".
        r = self.server.call("load", {"model": os.path.join(self.tmp, "x\u0000y")}, id="nul")
        self.assertError(r, "invalid_params", "nul")
        self.assertIn("NUL", r["error"]["message"])
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})

    def assert_protocol_fd_private(self, server):
        """Linux: exactly one fd of the server refers to its stdout pipe. It is >= 3 and close-on-exec, so neither
        fd 1 / fd 2 writes nor a child process can reach the protocol stream."""
        proc_fd = "/proc/%d/fd" % server.proc.pid
        if not os.path.isdir(proc_fd):
            return
        pipe = "pipe:[%d]" % os.fstat(server.proc.stdout.fileno()).st_ino
        fds = sorted(int(fd) for fd in os.listdir(proc_fd) if os.readlink(os.path.join(proc_fd, fd)) == pipe)
        self.assertEqual(len(fds), 1, "fds on the protocol pipe: %r" % fds)
        self.assertGreaterEqual(fds[0], 3)
        with open("/proc/%d/fdinfo/%d" % (server.proc.pid, fds[0])) as f:
            flags = int(next(l for l in f if l.startswith("flags:")).split()[1], 8)
        self.assertTrue(flags & 0o2000000, "protocol fd %d is not O_CLOEXEC (flags %o)" % (fds[0], flags))

    def run_session(self, server):
        """hello, status, shutdown on `server`; the protocol stream must stay clean."""
        try:
            self.assertEqual(server.call("hello", {"protocol_version": 1})["result"]["protocol_version"], 1)
            self.assert_protocol_fd_private(server)
            self.assertEqual(server.call("load", {"model": os.path.join(self.tmp, "missing")})["error"]["code"],
                             "invalid_argument")
            self.assertEqual(server.call("status")["result"], {"state": "unloaded"})
            self.assertEqual(server.call("shutdown", id="bye"), {"id": "bye", "result": {}})
            self.assertEqual(server.proc.wait(timeout=TIMEOUT), 0)
        finally:
            server.close()
        self.assertEqual(server.bad_lines, [], "stdout carried non-protocol lines")

    def test_stderr_to_devnull(self):
        self.run_session(Server(stderr=subprocess.DEVNULL))

    def test_stderr_to_a_file(self):
        self.run_session(Server())

    def test_stderr_merged_into_stdout(self):
        # 2>&1: stderr is the protocol stream itself, so the server silences it rather than corrupting the stream.
        self.run_session(Server(stderr=subprocess.STDOUT))

    @unittest.skipUnless(POSIX, "closing an inherited fd before exec is POSIX-only")
    def test_stderr_closed(self):
        # Without fd 2 the protocol copy of stdout must not become fd 2 (stderr writes would land in the stream).
        self.run_session(Server(stderr=subprocess.DEVNULL, preexec_fn=lambda: os.close(2)))

    @unittest.skipUnless(POSIX, "closing an inherited fd before exec is POSIX-only")
    def test_stdin_closed(self):
        out = subprocess.run([SERVER], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=lambda: os.close(0),
                             timeout=TIMEOUT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, b"")

    @unittest.skipUnless(POSIX, "the test watches the process with POSIX calls")
    def test_parent_pid_exits_when_the_parent_dies(self):
        # The parent spawns the server with the test's stdin pipe, so the server sees no end of input when the parent
        # is killed: only the watchdog can stop it.
        parent_code = textwrap.dedent("""
            import os, subprocess, sys, time
            p = subprocess.Popen([sys.argv[1], "--parent-pid", str(os.getpid())], stdin=sys.stdin,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(p.pid, flush=True)
            time.sleep(600)
        """)
        parent = subprocess.Popen([sys.executable, "-I", "-c", parent_code, SERVER], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)
        try:
            pid = int(parent.stdout.readline())
            time.sleep(0.5)
            self.assertFalse(process_gone(pid), "moss-server exited while its parent was alive")
            parent.kill()
            parent.wait()
            self.assertTrue(wait_gone(pid, 10), "moss-server outlived its parent")
        finally:
            parent.kill()
            parent.wait()
            parent.stdin.close()
            parent.stdout.close()

    @unittest.skipUnless(POSIX, "the test watches the process with POSIX calls")
    def test_parent_pid_of_another_process(self):
        # --parent-pid names a process that is not the server's parent (a launcher using a helper to spawn).
        sleeper = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(600)"])
        server = Server(args=["--parent-pid", str(sleeper.pid)], stderr=subprocess.DEVNULL)
        try:
            self.assertEqual(server.call("status")["result"], {"state": "unloaded"})
            sleeper.kill()
            sleeper.wait()
            self.assertEqual(server.proc.wait(timeout=10), 3)
        finally:
            sleeper.kill()
            sleeper.wait()
            server.close()
        # A process that is already gone is refused at startup.
        out = subprocess.run([SERVER, "--parent-pid", str(sleeper.pid)], stdin=subprocess.DEVNULL,
                             capture_output=True, timeout=TIMEOUT)
        self.assertEqual(out.returncode, 1, out.stderr)
        self.assertIn(b"not running", out.stderr)


@unittest.skipUnless(GPU, "set MOSSINFER_GPU_TESTS=1 (and hold the GPU lock) for the GPU tests")
class GpuProtocolTest(ProtocolTestBase):
    def load_base(self):
        base = os.path.join(WEIGHTS, "base")
        if not os.path.isdir(os.path.join(base, "transformer")):
            self.skipTest("weights not present: " + base)
        r = self.server.call("load", {"model": base, "mode": "balanced"})
        self.assertIn("result", r, r)
        self.assertEqual(r["result"]["model"], base)
        return base

    def wait_done(self, gen_id, events, timeout=GPU_TIMEOUT):
        deadline = time.monotonic() + timeout
        while True:
            msg = self.server.read(max(0.1, deadline - time.monotonic()))
            self.assertIn("event", msg, msg)
            events.append(msg)
            if msg["event"] == "done" and msg["id"] == gen_id:
                return msg

    def test_generate_256_two_steps(self):
        base = self.load_base()
        out = os.path.join(self.tmp, "apple.png")
        events = []
        gid = self.server.send("generate", {"prompt": "a red apple on a wooden table", "negative_prompt": "blurry",
                                            "width": 256, "height": 256, "steps": 2, "seed": 7,
                                            "uncond_skip_layers": "22", "output": out}, id="g1")
        ack = self.server.response(gid, events=events)
        self.assertEqual(ack["result"], {"accepted": True})
        # One generation at a time.
        busy = self.server.send("generate", {"prompt": "x", "output": out}, id="g2")
        self.assertError(self.server.response(busy, events=events), "busy", "g2")
        st = self.server.call("status")["result"]
        self.assertIn(st["state"], ("generating", "idle"))
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["image"], out)
        with open(out, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        report = done["report"]
        self.assertEqual(report["image"], out)
        self.assertEqual(report["request"]["model"], base)
        self.assertEqual(report["request"]["seed"], 7)
        self.assertEqual(report["request"]["uncond_skip_layers"], [22])
        self.assertEqual(report["seed_used"], 7)
        self.assertEqual(report["steps_run"], 2)
        self.assertFalse(report["cancelled"])
        self.assertTrue(report["decode"]["ran"])
        for key in ("plan", "timings_s", "memory_gib", "host_memory_gib", "embedding_cache"):
            self.assertIn(key, report)
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("plan"), 1)
        plan = next(e for e in events if e["event"] == "plan")
        self.assertTrue(plan["explanation"])
        self.assertIsInstance(plan["plan"], dict)
        progress = [(e["phase"], e["step"]) for e in events if e["event"] == "progress"]
        self.assertEqual(progress, [("text_encoder", 0), ("dit_load", 0), ("denoise", 0), ("denoise", 1), ("denoise", 2),
                                    ("decode", 0)])
        self.assertTrue(all(e["total_steps"] == 2 for e in events if e["event"] == "progress"))
        self.assertEqual(self.server.call("status")["result"]["state"], "idle")
        # Invalid requests are rejected up front with the engine's wording.
        r = self.server.call("generate", {"prompt": "x", "width": 1000, "output": out}, id="bad")
        self.assertError(r, "invalid_params", "bad")
        self.assertIn("multiples of 16", r["error"]["message"])
        r = self.server.call("generate", {"prompt": "a red apple\u0000 and a pear", "output": out}, id="nul")
        self.assertError(r, "invalid_params", "nul")
        self.assertIn("NUL", r["error"]["message"])
        self.assertEqual(self.server.call("shutdown")["result"], {})
        self.assertEqual(self.server.proc.wait(timeout=TIMEOUT), 0)

    def test_cancel_mid_run(self):
        self.load_base()
        out = os.path.join(self.tmp, "never.png")
        events = []
        gid = self.server.send("generate", {"prompt": "a lighthouse at dusk", "width": 256, "height": 256,
                                            "steps": 50, "output": out}, id="long")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        while not any(e["event"] == "progress" and e["phase"] == "denoise" and e["step"] >= 2 for e in events):
            events.append(self.server.read(GPU_TIMEOUT))
        cid = self.server.send("cancel", {"id": "other"})
        self.assertEqual(self.server.response(cid, events=events)["result"], {"cancelling": False, "generating_id": "long"})
        cid = self.server.send("cancel", {"id": "long"})
        self.assertEqual(self.server.response(cid, events=events)["result"], {"cancelling": True, "generating_id": "long"})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "cancelled", done)
        self.assertGreaterEqual(done["steps_run"], 2)
        self.assertLess(done["steps_run"], 50)
        self.assertTrue(done["report"]["cancelled"])
        self.assertFalse(os.path.exists(out))
        # The engine stays usable; the cancel does not carry over.
        out2 = os.path.join(self.tmp, "after.png")
        gid = self.server.send("generate", {"prompt": "a lighthouse at dusk", "width": 256, "height": 256, "steps": 2,
                                            "output": out2}, id="after")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        self.assertEqual(self.wait_done(gid, events)["status"], "ok")
        self.assertTrue(os.path.isfile(out2))
        # A generation running at shutdown is cancelled; the server still exits cleanly.
        gid = self.server.send("generate", {"prompt": "a lighthouse at dusk", "width": 256, "height": 256, "steps": 50,
                                            "output": out}, id="late")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        sid = self.server.send("shutdown")
        msgs = []
        while True:
            m = self.server.read(GPU_TIMEOUT)
            msgs.append(m)
            if m.get("id") == sid and "event" not in m:
                break
        self.assertEqual(msgs[-1]["result"], {})
        late = [m for m in msgs if m.get("event") == "done" and m["id"] == "late"]
        self.assertEqual(len(late), 1)
        self.assertEqual(late[0]["status"], "cancelled")
        self.assertEqual(self.server.proc.wait(timeout=TIMEOUT), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
