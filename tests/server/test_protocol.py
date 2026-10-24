#!/usr/bin/env python3
"""moss-server protocol tests (docs/server-protocol.md). Standard library only.

    MOSS_SERVER=build/moss-server python3 tests/server/test_protocol.py
    MOSSINFER_GPU_TESTS=1 MOSS_SERVER=... python3 tests/server/test_protocol.py   # + generate / cancel on the GPU

The CPU tests never create an engine on a real bundle, so they run on machines without a GPU or weights. The GPU tests
need the base bundle under $MOSS_WEIGHTS (default /workspace/Models/MOSS-Image) and should run under the GPU lock.
"""
import json
import struct
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
import zlib

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


def write_png(path, width, height, pixel):
    """An 8-bit RGB PNG (color type 2) whose pixel (x, y) is pixel(x, y) -> (r, g, b)."""
    raw = b"".join(b"\0" + bytes(c for x in range(width) for c in pixel(x, y)) for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_dataspace(path, codes):
    """A dataspace.safetensors: an identity 2x2 latent_mix.Q and F32 step codes."""
    header = json.dumps({"latent_mix.Q": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                         "step_codes": {"dtype": "F32", "shape": [len(codes)], "data_offsets": [16, 16 + 4 * len(codes)]}})
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header.encode() + struct.pack("<4f", 1, 0, 0, 1) +
                struct.pack("<%df" % len(codes), *codes))


def png_chunks(path):
    """(type, data) of every chunk of the PNG at `path`."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG: " + path
    pos, chunks = 8, []
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        chunks.append((data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]))
        pos += 12 + n
    return chunks


def png_size(path):
    ihdr = png_chunks(path)[0][1]
    return int.from_bytes(ihdr[0:4], "big"), int.from_bytes(ihdr[4:8], "big")


def png_text(path):
    """keyword -> text of the tEXt and uncompressed iTXt chunks."""
    out = {}
    for kind, body in png_chunks(path):
        if kind == b"tEXt":
            key, _, text = body.partition(b"\0")
            out[key.decode("latin-1")] = text.decode("latin-1")
        elif kind == b"iTXt":
            key, _, rest = body.partition(b"\0")
            if rest[0] == 0:  # uncompressed: flag, method, language\0, translated keyword\0, text
                rest = rest[2:].split(b"\0", 2)[2]
                out[key.decode("latin-1")] = rest.decode("utf-8")
    return out


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
        self.assertEqual(res["capabilities"], {"preview": True, "comfyui_models": False, "model_files": True,
                                               "lora": True, "tiers": True, "samplers": True,
                                               "image_input": True, "hires": True, "distill": True,
                                               "system_cuda": True, "control": True, "accel": True,
                                               "preload": True, "release": True, "certificate_dirs": True,
                                               "te_host_pin": True, "progress_fraction": True})
        probe = res["probe"]
        self.assertGreaterEqual(probe["device_count"], probe["supported_device_count"])
        self.assertIn(probe["recommended_flavour"], ("cu12", "cu13", ""))
        self.assertIn(probe["build_flavour"], ("cu12", "cu13"))
        self.assertTrue(probe["message"])
        self.assertNotIn("devices", probe)
        lt = probe["cublaslt"]
        self.assertEqual(set(lt), {"source", "version", "path", "error"})
        if lt["source"]:
            self.assertIn(lt["source"], ("system", "bundled", "env"))
            self.assertEqual(lt["version"] // 10000, {"cu12": 12, "cu13": 13}[probe["build_flavour"]])
            self.assertTrue(os.path.isfile(lt["path"]), lt)
            self.assertEqual(lt["error"], "")
        else:
            self.assertEqual((lt["version"], lt["path"]), (0, ""))
            self.assertTrue(lt["error"].startswith("no usable cuBLASLt"), lt["error"])
            self.assertRegex(lt["error"], r"rejected:|missing \(|no candidates")
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
        r = self.server.call("load", {"model": "/x", "block_path": "turbo"}, id=14)
        self.assertError(r, "invalid_params", 14)
        self.assertIn("block_path", r["error"]["message"])
        self.assertError(self.server.call("load", {"model": "/x", "block_path": 2}, id=15), "invalid_params", 15)
        # Every valid value (and null, the default) passes the server and reaches the engine's own checks.
        missing = os.path.join(self.tmp, "missing")
        for i, path in enumerate(("default", "op_by_op", "fused", "fused_graphs", None)):
            r = self.server.call("load", {"model": missing, "block_path": path}, id=20 + i)
            self.assertError(r, "invalid_argument", 20 + i)
            self.assertIn("not found", r["error"]["message"])
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})

    def test_p6_load_params(self):
        missing = os.path.join(self.tmp, "missing")
        bad = [({"tier": "fast"}, "tier"), ({"tier": 1}, "tier"), ({"attention": "q8"}, "attention"),
               ({"attention": "auto"}, "attention"), ({"lora_strict": 1}, "lora_strict"), ({"dit_file": 5}, "dit_file"),
               ({"text_encoder_file": ["x"]}, "text_encoder_file"), ({"vae_file": "a\u0000b"}, "vae_file")]
        for i, (params, name) in enumerate(bad):
            r = self.server.call("load", dict(params, model=missing), id=i)
            self.assertError(r, "invalid_params", i)
            self.assertIn(name, r["error"]["message"])
        # Every name passes the server and reaches the engine's own checks (here: the missing bundle).
        for i, tier in enumerate(("auto", "reference", "high", "turbo")):
            r = self.server.call("load", {"model": missing, "tier": tier}, id="t%d" % i)
            self.assertError(r, "invalid_argument", "t%d" % i)
        for i, att in enumerate(("default", "exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc")):
            r = self.server.call("load", {"model": missing, "attention": att, "lora_strict": True}, id="a%d" % i)
            self.assertError(r, "invalid_argument", "a%d" % i)
            self.assertIn("not found", r["error"]["message"])
        # Component files must exist (checked before CUDA); "" means the bundle's.
        bundle = self.fake_bundle()
        for key in ("dit_file", "text_encoder_file", "vae_file"):
            r = self.server.call("load", {"model": bundle, key: missing}, id=key)
            self.assertError(r, "invalid_argument", key)
            self.assertIn(key, r["error"]["message"])
        r = self.server.call("load", {"model": bundle, "dit_file": "", "vram_budget_bytes": 1000}, id="empty")
        self.assertError(r, "invalid_argument", "empty")
        self.assertIn("vram_budget_bytes", r["error"]["message"])
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})

    def test_p6_generate_params(self):
        # The request is checked before the server's state: no engine is needed for these.
        lora = os.path.join(self.tmp, "style.safetensors")
        open(lora, "wb").close()
        base = {"prompt": "a cat", "output": os.path.join(self.tmp, "x.png")}
        bad = [
            ({"sampler": "euler_a"}, "sampler"), ({"sampler": 2}, "sampler"), ({"scheduler": "Karras"}, "scheduler"),
            ({"noise_layout": "batch"}, "noise_layout"), ({"noise_source": "cpu"}, "noise_source"),
            ({"flow_multiplier": "1"}, "flow_multiplier"), ({"flow_multiplier": 0}, "Flow multiplier"),
            ({"eta": -1}, "eta"), ({"s_noise": True}, "s_noise"), ({"dataspace": 3}, "dataspace"),
            ({"sampler": "dataspace"}, "dataspace"),
            ({"sampler": "dataspace", "dataspace": os.path.join(self.tmp, "missing.safetensors")}, "not found"),
            ({"loras": {"path": lora}}, "loras"), ({"loras": [lora]}, "loras[0]"),
            ({"loras": [{"strength": 1}]}, "loras[0].path"), ({"loras": [{"path": ""}]}, "loras[0].path"),
            ({"loras": [{"path": lora, "scale": 1}]}, "scale"), ({"loras": [{"path": lora, "strength": "1"}]}, "strength"),
            ({"loras": [{"path": lora}, {"path": os.path.join(self.tmp, "missing.safetensors")}]}, "loras[1].path"),
            ({"loras": [{"path": lora, "strength": 1e300}]}, "strength"),
            ({"save_metadata": 1}, "save_metadata"), ({"include_prompts": "no"}, "include_prompts"),
            ({"preview_every": 2}, "preview_output"), ({"preview_output": ""}, "preview_output"),
            ({"preview_output": 5}, "preview_output"),
            ({"preview_output": os.path.join(self.tmp, "p.png"), "preview_every": 0}, "preview_every"),
        ]
        for i, (params, text) in enumerate(bad):
            r = self.server.call("generate", dict(base, **params), id=i)
            self.assertError(r, "invalid_params", i)
            self.assertIn(text, r["error"]["message"], params)
        # Every name of every enum, and the other new params, pass validation: the answer is then no_engine.
        names = {"sampler": ["euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde",
                             "res_multistep"],
                 "scheduler": ["moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"],
                 "noise_layout": ["official", "batch_stream"], "noise_source": ["philox", "philox_device", "brownian_tree"]}
        for key, values in names.items():
            for v in values:
                self.assertError(self.server.call("generate", dict(base, **{key: v}), id=v), "no_engine", v)
        ds = os.path.join(self.tmp, "dataspace.safetensors")
        write_dataspace(ds, [1000, 750, 500, 250])  # read by the pre-check: the step codes bound the steps
        ok = dict(base, sampler="dataspace", dataspace=ds, loras=[{"path": lora, "strength": 0.5}, {"path": lora}],
                  flow_multiplier=1.5, eta=0.5, s_noise=1.0, save_metadata=False, include_prompts=False,
                  preview_output=os.path.join(self.tmp, "p.png"), preview_every=2)
        self.assertError(self.server.call("generate", ok, id="ok"), "no_engine", "ok")
        self.assertError(self.server.call("generate", dict(base, loras=[]), id="none"), "no_engine", "none")
        # The dataspace errors name the protocol's parameter, not the C field.
        for i, extra in enumerate(({}, {"dataspace": os.path.join(self.tmp, "missing.safetensors")})):
            r = self.server.call("generate", dict(base, sampler="dataspace", **extra), id="ds%d" % i)
            self.assertError(r, "invalid_params", "ds%d" % i)
            self.assertIn("dataspace", r["error"]["message"])
            self.assertNotIn("dataspace_path", r["error"]["message"])
        # Repeated members are rejected, in a LoRA entry and among the params (raw lines: json.dumps cannot repeat).
        out = json.dumps(base["output"])
        for i, params in enumerate(('"loras": [{"path": %s, "path": %s}]' % (json.dumps(lora), json.dumps(lora)),
                                    '"loras": [{"path": %s, "strength": 1, "strength": 2}]' % json.dumps(lora),
                                    '"steps": 2, "steps": 3', '"sampler": "heun", "sampler": "euler"')):
            line = '{"id": "dup%d", "method": "generate", "params": {"prompt": "a cat", "output": %s, %s}}\n' % (i, out, params)
            self.server.send_raw(line.encode("utf-8"))
            r = self.server.read()
            self.assertError(r, "invalid_params", "dup%d" % i)
            self.assertIn("more than once", r["error"]["message"])

    def test_p7b_load_params(self):
        bundle = self.fake_bundle()
        missing = os.path.join(self.tmp, "missing")
        for i, params in enumerate(({"time_embedding_file": 5}, {"distill": ["x"]}, {"distill": "a\u0000b"},
                                    {"distill": ""}, {"time_embedding_file": ""})):
            self.assertError(self.server.call("load", dict(params, model=bundle), id=i), "invalid_params", i)
        r = self.server.call("load", {"model": bundle, "time_embedding_file": missing}, id="te")
        self.assertError(r, "invalid_argument", "te")
        self.assertIn("time_embedding_file", r["error"]["message"])
        r = self.server.call("load", {"model": bundle, "distill": missing}, id="d")
        self.assertError(r, "invalid_argument", "d")
        self.assertIn("distill: not a directory", r["error"]["message"])
        empty = os.path.join(self.tmp, "empty_bundle")
        os.makedirs(empty)
        self.assertError(self.server.call("load", {"model": bundle, "distill": empty}, id="e"), "model", "e")
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})

    def test_p7b_generate_params(self):
        png = os.path.join(self.tmp, "init.png")
        write_png(png, 32, 16, lambda x, y: (x * 8, y * 16, 100))
        mask = os.path.join(self.tmp, "mask.png")
        write_png(mask, 32, 16, lambda x, y: (255, 255, 255) if x < 16 else (0, 0, 0))
        mask01 = os.path.join(self.tmp, "mask01.png")
        write_png(mask01, 32, 16, lambda x, y: (1, 1, 1) if x < 16 else (0, 0, 0))
        missing = os.path.join(self.tmp, "missing.png")
        dmd = os.path.join(self.tmp, "dmd")
        os.makedirs(dmd)
        write_dataspace(os.path.join(dmd, "dataspace.safetensors"), [1000, 750, 500, 250])
        base = {"prompt": "a cat", "output": os.path.join(self.tmp, "x.png")}
        img = dict(base, init_image=png, denoise_strength=0.5)
        bad = [
            ({"init_image": 5}, "init_image"), (dict(img, init_image=missing), "init_image: file not found"),
            ({"denoise_strength": 0.5}, "needs an init image"), (dict(img, denoise_strength="0.5"), "denoise_strength"),
            (dict(img, denoise_strength=2), "Denoise strength"), ({"mask": mask}, "needs an init image"),
            (dict(img, mask=missing), "mask: file not found"),
            ({"steps": 0}, "steps"), ({"cfg": -1}, "cfg"), (dict(img, denoise_strength=-0.1), "denoise_strength"),
            ({"init_image": ""}, "non-empty path"), (dict(img, mask=""), "non-empty path"), ({"distill": ""}, "non-empty path"),
            ({"hires": {"width": 0, "height": 384}}, "hires.width"), ({"hires": {"width": 384, "height": 0, "steps": 2}}, "hires.height"),
            (dict(img, mask_encoding="bin"), "mask_encoding"), (dict(img, resize="cover"), "resize"),
            ({"hires": [384, 384]}, "hires"), ({"hires": {"width": 384}}, "needs \"width\" and \"height\""),
            ({"hires": {"width": 384, "height": 384, "scale": 2}}, "unknown member"),
            ({"hires": {"width": "384", "height": 384}}, "hires.width"), ({"hires": {"width": 380, "height": 384}}, "multiples of 16"),
            ({"hires": {"width": 384, "height": 384, "strength": "x"}}, "hires.strength"),
            ({"hires": {"width": 384, "height": 384, "strength": 2}}, "denoise strength"),
            ({"distill": 3}, "distill"), ({"distill": missing}, "distill: not a directory"),
            (dict(img, distill=dmd), "distill selected the bundle's dataspace sampler"),
        ]
        for i, (params, text) in enumerate(bad):
            r = self.server.call("generate", dict(base, **params), id=i)
            self.assertError(r, "invalid_params", i)
            self.assertIn(text, r["error"]["message"], params)
            for c_name in ("init_image_path", "mask_path", "distill_dir", "hires_width"):
                self.assertNotIn(c_name, r["error"]["message"], params)
        # Repeated hires members (a raw line: json.dumps cannot repeat).
        line = ('{"id": "dup", "method": "generate", "params": {"prompt": "a cat", "output": %s, '
                '"hires": {"width": 384, "width": 512, "height": 384}}}\n' % json.dumps(base["output"]))
        self.server.send_raw(line.encode("utf-8"))
        r = self.server.read()
        self.assertError(r, "invalid_params", "dup")
        self.assertIn("more than once", r["error"]["message"])
        # Valid requests reach the server's state check.
        ok = [dict(img, mask=mask, resize="fit"), dict(img, mask=mask01, mask_encoding="binary01", resize="stretch"),
              dict(img, mask=mask, mask_encoding="u8", resize="crop", denoise_strength=0),
              dict(base, hires={"width": 384, "height": 384}), dict(img, hires={"width": 384, "height": 384, "strength": 0.4, "steps": 2}),
              dict(base, distill=dmd), dict(img, distill=dmd, sampler="euler_ancestral"),
              dict(base, hires={"width": 384, "height": 384, "steps": 0}),
              # Files are decoded at generate time, not by the pre-check: a binary01 mask with other values and an
              # undecodable init image are accepted here (and fail in "done").
              dict(img, mask=mask, mask_encoding="binary01"), dict(img, init_image=os.path.join(self.tmp, "x.png"))]
        open(os.path.join(self.tmp, "x.png"), "wb").write(b"not a png")
        for i, params in enumerate(ok):
            self.assertError(self.server.call("generate", params, id="ok%d" % i), "no_engine", "ok%d" % i)

    def test_control_and_accelerator_params(self):
        adapter = os.path.join(self.tmp, "control.safetensors")
        with open(adapter, "wb") as f:
            f.write(b"read by generate, not by the pre-check")
        depth = os.path.join(self.tmp, "depth.png")
        write_png(depth, 32, 16, lambda x, y: (x * 8, x * 8, x * 8))
        missing = os.path.join(self.tmp, "missing.png")
        base = {"prompt": "a cat", "output": os.path.join(self.tmp, "x.png")}
        ctl = {"adapter": adapter, "images": [{"kind": "depth", "path": depth}]}
        bad = [
            ({"control": adapter}, '"control" must be an object'),
            ({"control": {}}, '"control.adapter" (the control adapter file) is required'),
            ({"control": {"adapter": ""}}, '"control.adapter" must be a non-empty string'),
            ({"control": {"adapter": 5}}, '"control.adapter" must be a non-empty string'),
            ({"control": {"adapter": adapter, "scale": 1}}, '"control" has an unknown member "scale"'),
            ({"control": {"adapter": adapter, "lora_strength": "1"}}, '"control.lora_strength" must be a number'),
            ({"control": {"adapter": adapter, "images": {}}}, '"control.images" must be an array'),
            ({"control": {"adapter": adapter, "images": ["x"]}}, '"control.images[0]" must be an object'),
            ({"control": {"adapter": adapter, "images": [{"path": depth}]}}, '"control.images[0].kind" is required'),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth"}]}}, '"control.images[0].path" (the image file) is required'),
            ({"control": {"adapter": adapter, "images": [{"kind": "", "path": depth}]}}, '"control.images[0].kind" must be'),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": ""}]}}, '"control.images[0].path" must be'),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": depth, "image": depth}]}},
             '"control.images[0]" has an unknown member "image"'),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": depth, "strength": "2"}]}},
             '"control.images[0].strength" must be a number'),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": depth, "start": None}]}},
             '"control.images[0].start" must be a number'),
            ({"control": {"adapter": adapter + ".missing"}}, "control.adapter: file not found"),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": missing}]}},
             "control.images[0].path: file not found"),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": depth, "strength": 11}]}},
             "control.images[0].strength must be between 0 and 10"),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": depth, "start": 0.8, "end": 0.2}]}},
             "control.images[0]: require 0 <= start <= end <= 1"),
            ({"control": dict(ctl, images=ctl["images"] * 2)}, 'Two control images of kind "depth"'),
            ({"control": dict(ctl, images=ctl["images"] * 17)}, "control.images: at most 16 entries"),
            ({"control": dict(ctl, lora_strength=1e300)}, "control.lora_strength must be a finite number"),
            ({"cfg_truncation": "0.5"}, '"cfg_truncation" must be a number'),
            ({"cfg_truncation": 1.5}, "cfg_truncation must be between 0 and 1"),
            ({"cfg_truncation": -0.1}, "cfg_truncation must be between 0 and 1"),
            ({"step_cache": "teacache"}, '"step_cache" must be "off" or "fbcache"'),
            ({"step_cache": True}, '"step_cache" must be a string'),
            ({"step_cache": "fbcache", "step_cache_threshold": 0}, '"step_cache_threshold" must be a number in (0, 1]'),
            ({"step_cache": "fbcache", "step_cache_threshold": 1.5}, '"step_cache_threshold" must be a number in (0, 1]'),
            ({"step_cache_threshold": "0.05"}, '"step_cache_threshold" must be a number'),
        ]
        for i, (params, text) in enumerate(bad):
            r = self.server.call("generate", dict(base, **params), id=i)
            self.assertError(r, "invalid_params", i)
            self.assertIn(text, r["error"]["message"], params)
            for c_name in ("control_adapter", "control_images", "image_path", "control_lora_strength", "mi_"):
                self.assertNotIn(c_name, r["error"]["message"], params)
        # File paths that contain C field names come back unchanged; the field names around them are still renamed.
        os.makedirs(os.path.join(self.tmp, "control_images"))
        paths = {name: os.path.join(self.tmp, name) for name in (
            "my_control_adapter.safetensors", "x_image_path.png", os.path.join("control_images", "a.png"),
            "my_mask_path.png", "my_distill_dir", "x_dataspace_path.safetensors", "control_adapter")}
        cases = [
            ({"control": {"adapter": paths["my_control_adapter.safetensors"]}}, "control.adapter: file not found: ",
             paths["my_control_adapter.safetensors"]),
            ({"control": {"adapter": paths["control_adapter"]}}, "control.adapter: file not found: ", paths["control_adapter"]),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": paths["x_image_path.png"]}]}},
             "control.images[0].path: file not found: ", paths["x_image_path.png"]),
            ({"control": {"adapter": adapter, "images": [{"kind": "depth", "path": paths[os.path.join("control_images", "a.png")]}]}},
             "control.images[0].path: file not found: ", paths[os.path.join("control_images", "a.png")]),
            ({"init_image": depth, "mask": paths["my_mask_path.png"]}, "mask: file not found: ", paths["my_mask_path.png"]),
            ({"distill": paths["my_distill_dir"]}, "distill: not a directory: ", paths["my_distill_dir"]),
            ({"sampler": "dataspace", "dataspace": paths["x_dataspace_path.safetensors"]}, "dataspace: file not found: ",
             paths["x_dataspace_path.safetensors"]),
        ]
        for i, (params, field, path) in enumerate(cases):
            r = self.server.call("generate", dict(base, **params), id="p%d" % i)
            self.assertError(r, "invalid_params", "p%d" % i)
            self.assertIn(field + path, r["error"]["message"], params)
        # Repeated members (raw lines: json.dumps cannot repeat).
        for gid, control in (("dup0", '{"adapter": %s, "adapter": %s}' % (json.dumps(adapter), json.dumps(adapter))),
                             ("dup1", '{"adapter": %s, "images": [{"kind": "depth", "kind": "color", "path": %s}]}'
                              % (json.dumps(adapter), json.dumps(depth)))):
            line = ('{"id": "%s", "method": "generate", "params": {"prompt": "a cat", "output": %s, "control": %s}}\n'
                    % (gid, json.dumps(base["output"]), control))
            self.server.send_raw(line.encode("utf-8"))
            r = self.server.read()
            self.assertError(r, "invalid_params", gid)
            self.assertIn("more than once", r["error"]["message"])
        # Valid requests reach the server's state check: the adapter's format and kinds, and the images' contents,
        # are checked at generate time.
        ok = [dict(base, control=ctl), dict(base, control={"adapter": adapter}),
              dict(base, control={"adapter": adapter, "lora_strength": 0.5, "images": [
                  {"kind": "depth", "path": depth, "strength": 0.5, "start": 0, "end": 0.6},
                  {"kind": "color", "path": depth, "strength": 2}]}),
              dict(base, control={"adapter": adapter, "images": [{"kind": "canny", "path": adapter}]}),
              dict(base, cfg_truncation=0.8), dict(base, step_cache="fbcache"), dict(base, step_cache="off"),
              dict(base, step_cache="fbcache", step_cache_threshold=0.05), dict(base, step_cache_threshold=1),
              dict(base, control=ctl, cfg_truncation=0, step_cache="fbcache", hires={"width": 384, "height": 384})]
        for i, params in enumerate(ok):
            self.assertError(self.server.call("generate", params, id="ok%d" % i), "no_engine", "ok%d" % i)

    def test_load_params_11(self):
        missing = os.path.join(self.tmp, "missing")
        bundle = self.fake_bundle()
        certs = os.path.join(self.tmp, "certs")
        bad = [({"certificate_dirs": certs}, '"certificate_dirs" must be an array'),
               ({"certificate_dirs": [certs, ""]}, '"certificate_dirs[1]" must be a non-empty path'),
               ({"certificate_dirs": [5]}, '"certificate_dirs[0]" must be a non-empty path'),
               ({"certificate_dirs": ["a\u0000b"]}, '"certificate_dirs[0]" must be a non-empty path without NUL'),
               ({"dit_direct_io": "on"}, '"dit_direct_io" must be "auto", "buffered" or "direct"'),
               ({"dit_direct_io": True}, '"dit_direct_io" must be a string'),
               ({"te_host_pin": "pinned"}, '"te_host_pin" must be "auto", "on" or "off"'),
               ({"te_host_pin": 1}, '"te_host_pin" must be a string')]
        for i, (params, message) in enumerate(bad):
            r = self.server.call("load", dict(params, model=missing), id=i)
            self.assertError(r, "invalid_params", i)
            self.assertIn(message, r["error"]["message"])
        # Every valid value passes the server and reaches the engine's own checks (here: the missing bundle); an
        # absent or null parameter is the default.
        for i, extra in enumerate(({"certificate_dirs": []}, {"certificate_dirs": [certs, missing]},
                                   {"certificate_dirs": None}, {"dit_direct_io": "auto"}, {"dit_direct_io": "buffered"},
                                   {"dit_direct_io": "direct", "direct_io": True}, {"te_host_pin": "auto"},
                                   {"te_host_pin": "on"}, {"te_host_pin": "off"}, {"dit_direct_io": None})):
            r = self.server.call("load", dict(extra, model=missing), id="ok%d" % i)
            self.assertError(r, "invalid_argument", "ok%d" % i)
            self.assertIn("not found", r["error"]["message"])
        # The engine counts the directories (at most 16; a missing one is searched as empty), before CUDA.
        r = self.server.call("load", {"model": bundle, "certificate_dirs": [certs] * 17}, id="many")
        self.assertError(r, "invalid_argument", "many")
        self.assertIn("certificate_dirs: at most 16 entries (got 17)", r["error"]["message"])
        r = self.server.call("load", {"model": bundle, "certificate_dirs": [certs] * 16, "dit_direct_io": "buffered",
                                      "te_host_pin": "off", "vram_budget_bytes": 1000}, id="16")
        self.assertError(r, "invalid_argument", "16")
        self.assertIn("vram_budget_bytes", r["error"]["message"])
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})

    def test_preload_and_release_params(self):
        lora = os.path.join(self.tmp, "style.safetensors")
        with open(lora, "wb") as f:
            f.write(b"read by preload, not by the pre-check")
        out = os.path.join(self.tmp, "x.png")
        # preload takes generate's request without the output, the preview and the PNG metadata.
        for i, key in enumerate(("output", "preview_output", "preview_every", "save_metadata", "include_prompts",
                                 "frobnicate")):
            r = self.server.call("preload", {key: out if "output" in key else True}, id="u%d" % i)
            self.assertError(r, "invalid_params", "u%d" % i)
            self.assertIn('unknown parameter "%s"' % key, r["error"]["message"])
        # The request is checked like generate's, before the server's state.
        bad = [({"width": 1000}, "multiples of 16"), ({"sampler": "euler_a"}, "sampler"), ({"steps": 0}, "steps"),
               ({"loras": [{"path": os.path.join(self.tmp, "none.safetensors")}]}, "file not found"),
               ({"step_cache_threshold": 0}, "step_cache_threshold"), ({"prompt": 5}, '"prompt" must be a string'),
               ({"control": {"adapter": os.path.join(self.tmp, "none.safetensors")}}, "control.adapter")]
        for i, (params, message) in enumerate(bad):
            r = self.server.call("preload", params, id="b%d" % i)
            self.assertError(r, "invalid_params", "b%d" % i)
            self.assertIn(message, r["error"]["message"])
        # No prompt, a blank one, or a full request: valid, then no engine.
        for i, params in enumerate((None, {}, {"prompt": ""}, {"prompt": "  "}, {"prompt": "a cat", "width": 512,
                                    "height": 768, "steps": 8, "cfg": 1, "loras": [{"path": lora, "strength": 0.5}],
                                    "step_cache": "fbcache"})):
            self.assertError(self.server.call("preload", params, id="p%d" % i), "no_engine", "p%d" % i)
        # generate still requires its prompt (through the engine's own check).
        r = self.server.call("generate", {"output": out}, id="g")
        self.assertError(r, "invalid_params", "g")
        self.assertIn("Enter a prompt.", r["error"]["message"])
        # release: keep_host_copy is a boolean (default true); nothing else.
        for i, params in enumerate(({"keep_host_copy": 1}, {"keep_host_copy": "yes"}, {"keep": True})):
            self.assertError(self.server.call("release", params, id="r%d" % i), "invalid_params", "r%d" % i)
        for i, params in enumerate((None, {}, {"keep_host_copy": True}, {"keep_host_copy": False},
                                    {"keep_host_copy": None})):
            self.assertError(self.server.call("release", params, id="n%d" % i), "no_engine", "n%d" % i)
        self.assertEqual(self.server.call("status")["result"], {"state": "unloaded"})
        self.assertEqual(self.server.call("cancel")["result"], {"cancelling": False})

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
        r = self.server.call("load", {"model": base, "mode": "balanced", "block_path": "fused"})
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

    def assertWholeJobProgress(self, progress):
        """fraction and eta_seconds in every progress event: the fraction in [0, 1) and never decreasing, the eta known
        (the plan comes first) and >= 0."""
        self.assertTrue(progress)
        last = 0.0
        for e in progress:
            self.assertIsInstance(e["fraction"], (int, float), e)
            self.assertGreaterEqual(e["fraction"], last, progress)
            self.assertLess(e["fraction"], 1.0, e)
            last = e["fraction"]
            self.assertIsNotNone(e["eta_seconds"], e)
            self.assertGreaterEqual(e["eta_seconds"], 0, e)

    def assertBarMoves(self, progress):
        """A generation's bar moves: from text_encoder to the final decode the fraction grows past one half and the eta
        shrinks."""
        first, last = progress[0], progress[-1]
        self.assertEqual((first["phase"], last["phase"]), ("text_encoder", "decode"), progress)
        self.assertGreater(last["fraction"], first["fraction"], progress)
        self.assertGreater(last["fraction"], 0.5, progress)
        self.assertLess(last["eta_seconds"], first["eta_seconds"], progress)

    def assertStatusProgress(self, st):
        """A running job's status: the progress fields, the whole-job ones included."""
        for key in ("phase", "step", "total_steps", "elapsed_seconds", "fraction", "eta_seconds"):
            self.assertIn(key, st)
        self.assertGreaterEqual(st["fraction"], 0)
        self.assertLess(st["fraction"], 1)
        self.assertTrue(st["eta_seconds"] is None or st["eta_seconds"] >= 0, st)

    def test_preload_release_generate(self):
        self.load_base()
        events = []
        request = {"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 2, "seed": 5}
        # preload: async like generate, with a plan, a progress and a done carrying dit_load and preload_note.
        pid = self.server.send("preload", request, id="p1")
        self.assertEqual(self.server.response(pid, events=events)["result"], {"accepted": True})
        busy = self.server.send("release", {}, id="busy")
        self.assertError(self.server.response(busy, events=events), "busy", "busy")
        done = self.wait_done(pid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertIn("preload_note", done)
        self.assertEqual(set(done["dit_load"]), {"source", "io", "bytes", "seconds", "note"})
        self.assertEqual(done["dit_load"]["source"], done["report"]["dit_load"]["source"])
        if not done["preload_note"]:
            self.assertEqual(done["dit_load"]["source"], "files")
            self.assertIn(done["dit_load"]["io"], ("direct", "buffered"))
        self.assertIsInstance(done["warnings"], list)
        # The plan before the first load step, then a progress (step 0) per load step, in order; never denoise.
        mine = [e for e in events if e.get("id") == "p1" and e["event"] in ("plan", "progress")]
        self.assertEqual(mine[0]["event"], "plan", mine)
        self.assertEqual([e["event"] for e in mine].count("plan"), 1)
        phases = [e["phase"] for e in mine[1:]]
        rank = {"text_encoder": 0, "dit_load": 1, "decode": 2}  # a txt2img request: decode is the VAE decoder
        self.assertTrue(phases, mine)
        self.assertEqual([rank[p] for p in phases], sorted(rank[p] for p in phases), phases)
        self.assertEqual(len(set(phases)), len(phases), phases)
        self.assertTrue(all(e["step"] == 0 and e["total_steps"] == 0 for e in mine[1:]), mine)
        self.assertWholeJobProgress(mine[1:])
        if not done["preload_note"]:
            self.assertIn("dit_load", phases)
        self.assertEqual(self.server.call("status")["result"]["state"], "idle")
        # release with the host copy: answered when it ends, with the stats.
        rid = self.server.send("release", {"keep_host_copy": True}, id="r1")
        r = self.server.response(rid, events=events, timeout=GPU_TIMEOUT)
        stats = r["result"]
        self.assertEqual(set(stats), {"cancelled", "seconds", "vram_released_bytes", "host_copy_bytes",
                                      "host_copy_seconds", "dit_host_copy", "te_host_copy", "note"})
        self.assertIs(stats["cancelled"], False)
        self.assertGreater(stats["vram_released_bytes"], 0)
        print("release:", stats, file=sys.stderr)
        self.assertEqual(self.server.call("status")["result"]["state"], "idle")  # the engine stays
        # generate: the DiT comes back from the host copy.
        out = os.path.join(self.tmp, "after_release.png")
        gid = self.server.send("generate", dict(request, output=out), id="g1")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["report"]["dit_load"]["source"], "host_copy" if stats["dit_host_copy"] else "files")
        progress = [e for e in events if e.get("id") == "g1" and e["event"] == "progress"]
        self.assertWholeJobProgress(progress)
        self.assertBarMoves(progress)
        # The host's room decides whether a copy is kept: without it the release says why, and the cancel below can
        # only find that the copy does not fit.
        no_room = not stats["dit_host_copy"]
        if no_room:
            self.assertIn("not kept in RAM", stats["note"], stats)
            print("NOTE: no DiT host copy on this machine now (%s); the host-copy load is not checked" % stats["note"],
                  file=sys.stderr)
        # A cancel sent right behind a release{keep_host_copy: true}. Landing before the release frees anything, it
        # stops it whole ("cancelled": true, nothing released); later, it stops at most the DiT's copy, and the release
        # still frees the device memory, keeps the text encoder's copy and frees nothing else.
        self.server.send("release", {"keep_host_copy": True}, id="r2")
        self.server.send("cancel", {"id": "r2"}, id="c2")
        answers = {}
        while len(answers) < 2:
            m = self.server.read(GPU_TIMEOUT)
            if "event" not in m:
                answers[m["id"]] = m
        self.assertIn("result", answers["r2"], answers)
        got = answers["r2"]["result"]
        print("release + cancel:", answers["c2"]["result"], got, file=sys.stderr)
        # The cancel found the release running or already done (False: it had ended), and the DiT copy was either
        # interrupted (no copy, the note says so) or finished before the cancel reached it.
        self.assertIn(answers["c2"]["result"]["cancelling"], (True, False))
        if got["cancelled"]:
            self.assertEqual(got["vram_released_bytes"], 0, got)
            self.assertEqual(got["note"], "cancelled before it started", got)
        elif got["dit_host_copy"]:
            self.assertEqual(got["note"], "", got)
        else:
            # Interrupted, or (the host's free memory changes) no room for the copy at all.
            self.assertTrue("cancelled" in got["note"] or "not kept in RAM" in got["note"], got)
        if not got["cancelled"]:
            self.assertGreater(got["vram_released_bytes"], 0)
        self.assertEqual(got["te_host_copy"], stats["te_host_copy"], got)  # the text encoder's copy is never dropped
        # The cancel does not carry over: the next generate runs every step (with no DiT load after an early cancel).
        gid = self.server.send("generate", dict(request, output=out), id="g2")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["report"]["steps_run"], 2)
        if got["cancelled"]:
            self.assertIsNone(done["report"]["dit_load"]["source"], done["report"]["dit_load"])
        self.assertEqual(self.server.call("unload")["result"], {"unloaded": True})

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
        if st["state"] == "generating":
            self.assertStatusProgress(st)
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["image"], out)
        self.assertIs(done["certified"], done["report"]["models"]["dit"]["certified"])
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
        for key in ("plan", "timings_s", "memory_gib", "host_memory_gib", "embedding_cache", "hardware"):
            self.assertIn(key, report)
        # The default block path is fused; op by op only after a failed self-check, and the note says why.
        blocks = report["dit_blocks"]
        self.assertIn(blocks["path"], ("fused", "op_by_op"), blocks)
        self.assertEqual(blocks["note"] == "", blocks["path"] == "fused", blocks)
        # 256^2, balanced: every DiT block resident on the dev box.
        self.assertFalse(report["dit_stream"]["streamed"], report["dit_stream"])
        self.assertEqual(report["dit_stream"]["steps"], [])
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("plan"), 1)
        plan = next(e for e in events if e["event"] == "plan")
        self.assertTrue(plan["explanation"])
        self.assertIsInstance(plan["plan"], dict)
        progress = [(e["phase"], e["step"]) for e in events if e["event"] == "progress"]
        self.assertEqual(progress, [("text_encoder", 0), ("dit_load", 0), ("denoise", 0), ("denoise", 1), ("denoise", 2),
                                    ("decode", 0)])
        self.assertTrue(all(e["total_steps"] == 2 for e in events if e["event"] == "progress"))
        self.assertWholeJobProgress([e for e in events if e["event"] == "progress"])
        self.assertBarMoves([e for e in events if e["event"] == "progress"])
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

    def test_p6_preview_sampler_metadata(self):
        self.load_base()
        out = os.path.join(self.tmp, "p6.png")
        preview = os.path.join(self.tmp, "preview.png")
        events = []
        # max_sequence_length 16 truncates the prompt (its chat template alone is longer): done.warnings says so.
        gid = self.server.send("generate", {"prompt": "a red apple on a wooden table, studio light, high detail",
                                            "width": 256, "height": 256, "steps": 3, "seed": 7,
                                            "sampler": "euler_ancestral", "scheduler": "karras", "eta": 1.0,
                                            "save_metadata": True, "max_sequence_length": 16,
                                            "preview_output": preview, "output": out}, id="p6")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        # While it runs, an invalid request still gets invalid_params (validation comes before busy).
        bad = self.server.send("generate", {"prompt": "x", "sampler": "euler_a", "output": out}, id="bad-while-busy")
        r = self.server.response(bad, events=events)
        self.assertError(r, "invalid_params", "bad-while-busy")
        self.assertIn("sampler", r["error"]["message"])
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        previews = [e for e in events if e["event"] == "preview"]
        self.assertEqual([e["step"] for e in previews], [1, 2, 3])
        for e in previews:
            self.assertEqual((e["id"], e["image"], e["width"], e["height"], e["total_steps"]), ("p6", preview, 256, 256, 3))
        # Each preview event follows its write and precedes that step's progress event.
        kinds = [(e["event"], e["step"]) for e in events
                 if e["event"] == "preview" or (e["event"] == "progress" and e["phase"] == "denoise")]
        self.assertEqual(kinds, [("progress", 0), ("preview", 1), ("progress", 1), ("preview", 2), ("progress", 2),
                                 ("preview", 3), ("progress", 3)])
        self.assertEqual(png_size(preview), (256, 256))
        self.assertNotIn("generation", png_text(preview))
        text = png_text(out)
        meta = json.loads(text["generation"])
        self.assertEqual((meta["sampler"], meta["scheduler"], meta["steps"], meta["seed"]), ("euler_ancestral", "karras", 3, 7))
        self.assertIn("prompt", meta)
        self.assertIn("parameters", text)
        report = done["report"]
        self.assertEqual(report["request"]["sampler"], "euler_ancestral")
        self.assertTrue(report["tokens"]["prompt"]["truncated"])
        self.assertIn("quality", report)
        self.assertIsInstance(done["warnings"], list)
        self.assertTrue(any(w.startswith("prompt truncated to 16 of ") for w in done["warnings"]), done["warnings"])
        # Without metadata or prompts.
        gid = self.server.send("generate", {"prompt": "a secret prompt", "width": 256, "height": 256, "steps": 2,
                                            "include_prompts": False, "output": out}, id="np")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["warnings"], [])
        text = png_text(out)
        self.assertNotIn("prompt", json.loads(text["generation"]))
        self.assertNotIn("a secret prompt", text["parameters"])
        gid = self.server.send("generate", {"prompt": "a cat", "width": 256, "height": 256, "steps": 2,
                                            "save_metadata": False, "output": out}, id="nm")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        self.assertEqual(self.wait_done(gid, events)["status"], "ok")
        self.assertEqual(png_text(out), {})

    def test_p7b_img2img_from_a_path(self):
        self.load_base()
        init = os.path.join(self.tmp, "init.png")
        write_png(init, 256, 256, lambda x, y: (x, y, 255 - x))
        out = os.path.join(self.tmp, "img2img.png")
        events = []
        gid = self.server.send("generate", {"prompt": "a red apple on a wooden table", "width": 256, "height": 256,
                                            "steps": 4, "init_image": init, "denoise_strength": 0.5, "resize": "crop",
                                            "output": out}, id="i2i")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(png_size(out), (256, 256))
        img = done["report"]["img2img"]
        self.assertEqual(img["mode"], "img2img")
        self.assertEqual(img["first_pass"]["start_step"], 2, img)
        self.assertEqual(done["report"]["steps_run"], 2)
        progress = [e for e in events if e["event"] == "progress"]
        self.assertTrue(all(e["total_steps"] == 2 for e in progress), progress)
        self.assertEqual(done["warnings"], [])
        # Strength 1 without a mask ignores the image: done.warnings carries the note.
        gid = self.server.send("generate", {"prompt": "a cat", "width": 256, "height": 256, "steps": 2,
                                            "init_image": init, "denoise_strength": 1, "output": out}, id="s1")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(done["report"]["img2img"]["mode"], "txt2img")
        self.assertTrue(any(w.startswith("image input: ") for w in done["warnings"]), done["warnings"])
        # Errors found during the generation name the protocol's parameters, never a C field.
        white = os.path.join(self.tmp, "white.png")
        write_png(white, 256, 256, lambda x, y: (255, 255, 255))
        dmd = os.path.join(self.tmp, "dmd")
        os.makedirs(dmd)
        write_dataspace(os.path.join(dmd, "dataspace.safetensors"), [1000, 750, 500, 250])
        open(os.path.join(dmd, "time_embedding.safetensors"), "wb").write(b"x")  # this engine has none
        for gid, params, text in (
                ("bin", {"init_image": init, "mask": white, "mask_encoding": "binary01"}, 'mask_encoding "binary01"'),
                ("te", {"distill": dmd}, "load's \"distill\""),
                ("dec", {"init_image": os.path.join(dmd, "time_embedding.safetensors")}, "init_image: ")):
            self.server.send("generate", dict({"prompt": "a cat", "width": 256, "height": 256, "output": out}, **params), id=gid)
            self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
            done = self.wait_done(gid, events)
            self.assertEqual(done["status"], "error", done)
            self.assertEqual(done["error"]["code"], "invalid_argument", done)
            message = done["error"]["message"]
            self.assertIn(text, message)
            for c_name in ("mi_", "MI_", "_dir", "_path", "init_image_path", "mask_path"):
                self.assertNotIn(c_name, message)

    def test_control_and_step_cache(self):
        community = os.environ.get("MOSS_COMMUNITY_MODELS") or "/workspace/Models/community"
        adapter = os.path.join(community, "comfy-org", "moss_image_color_depth_control.safetensors")
        if not os.path.isfile(adapter):
            self.skipTest("no control adapter: " + adapter + " (set MOSS_COMMUNITY_MODELS)")
        self.load_base()
        depth = os.path.join(self.tmp, "depth.png")
        write_png(depth, 256, 256, lambda x, y: (230,) * 3 if (x - 128) ** 2 + (y - 150) ** 2 < 70 ** 2 else (40 + y // 4,) * 3)
        out = os.path.join(self.tmp, "control.png")
        events = []
        gid = self.server.send("generate", {"prompt": "a red apple on a wooden table", "width": 256, "height": 256,
                                            "steps": 4, "output": out, "step_cache": "fbcache",
                                            "step_cache_threshold": 0.5, "control": {
                                                "adapter": adapter, "images": [{"kind": "depth", "path": depth}]}},
                               id="ctl")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertEqual(png_size(out), (256, 256))
        control = done["report"]["control"]
        self.assertEqual(control["adapter"], adapter)
        self.assertIn("depth", control["kinds"])
        self.assertGreater(control["encodes"], 0)
        accel = done["report"]["accelerators"]
        self.assertTrue(accel["lossy"], accel)
        self.assertGreater(accel["step_cache_calls"], 0, accel)
        self.assertFalse(done["report"]["models"]["dit"]["certified"], done["report"]["models"])
        self.assertIn("uncertified", done["report"]["quality"]["reason"])
        lossy = [w for w in done["warnings"] if w.startswith("lossy accelerators (")]
        self.assertEqual(len(lossy), 1, done["warnings"])
        self.assertIn("step_cache fbcache 0.5", lossy[0])
        self.assertIn("step cache reused", lossy[0])
        self.assertNotIn("CFG truncated", lossy[0])  # only the active accelerators
        self.assertIs(done["certified"], False)
        print("[server] control %s, %d encodes; %s" % (control["kinds"], control["encodes"], lossy[0]))
        # Settings without effect: a threshold without the step cache, CFG truncation at CFG 1. Nothing lossy ran, the
        # bf16 bundle ran Reference: certified.
        gid = self.server.send("generate", {"prompt": "a cat", "width": 256, "height": 256, "steps": 2, "cfg": 1,
                                            "cfg_truncation": 0.5, "step_cache_threshold": 0.1, "output": out}, id="noop")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "ok", done)
        self.assertIn('step_cache_threshold has no effect without step_cache "fbcache"', done["warnings"])
        self.assertIn("cfg_truncation has no effect without CFG (cfg 1)", done["warnings"])
        self.assertFalse(any(w.startswith("lossy accelerators") for w in done["warnings"]), done["warnings"])
        self.assertIs(done["certified"], True)
        self.assertIs(done["certified"], done["report"]["models"]["dit"]["certified"])
        # A kind the adapter does not have fails in "done", in the protocol's names.
        gid = self.server.send("generate", {"prompt": "a cat", "width": 256, "height": 256, "steps": 2, "output": out,
                                            "control": {"adapter": adapter, "images": [{"kind": "canny", "path": depth}]}},
                               id="kind")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        done = self.wait_done(gid, events)
        self.assertEqual(done["status"], "error", done)
        self.assertEqual(done["error"]["code"], "invalid_argument", done)
        self.assertIn('no "canny" input', done["error"]["message"])

    def test_cancel_mid_run(self):
        self.load_base()
        out = os.path.join(self.tmp, "never.png")
        events = []
        gid = self.server.send("generate", {"prompt": "a lighthouse at dusk", "width": 256, "height": 256,
                                            "steps": 50, "output": out}, id="long")
        self.assertEqual(self.server.response(gid, events=events)["result"], {"accepted": True})
        while not any(e["event"] == "progress" and e["phase"] == "denoise" and e["step"] >= 2 for e in events):
            events.append(self.server.read(GPU_TIMEOUT))
        # Mid-run, after measured steps: the status has the whole-job fields, the eta known.
        st = self.server.response(self.server.send("status"), events=events)["result"]
        self.assertEqual(st["state"], "generating", st)
        self.assertStatusProgress(st)
        self.assertIsNotNone(st["eta_seconds"], st)
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
