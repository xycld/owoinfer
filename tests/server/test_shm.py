#!/usr/bin/env python3
"""moss-server shared-memory regions (tools/server/shm.h) across processes. Standard library only.

    MOSS_SHM_HELPER=build/mossinfer_shm_helper python3 tests/server/test_shm.py

The helper (tests/server/shm_helper.cpp) stands in for the server; tests/server/shm_client.py is the client as an
integration attaches. Python clients: every interpreter in $MOSS_SHM_PYTHONS (os.pathsep-separated), else python3.10
and python3.13 from PATH where present, else this one.
"""
import os
import shutil
import subprocess
import sys
import time
import unittest

HELPER = os.environ.get("MOSS_SHM_HELPER", "")
CLIENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shm_client.py")
POSIX = os.name == "posix"
TIMEOUT = 30.0
MIB = 1 << 20
# This run's region names (the helper's --prefix): sweeps here never touch another test's or a real server's regions.
PREFIX = "mossinfer-t%d-" % os.getpid()


def pythons():
    env = os.environ.get("MOSS_SHM_PYTHONS")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    found = [p for p in (shutil.which("python3.10"), shutil.which("python3.13")) if p]
    return found or [sys.executable]


def python_version(exe):
    out = subprocess.run([exe, "-I", "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                         capture_output=True, text=True, timeout=TIMEOUT, check=True).stdout.split()
    return int(out[0]), int(out[1])


def shm_path(name):
    return "/dev/shm/" + name.lstrip("/")


class Helper:
    """A running shm_helper: one command line in, one answer line out."""

    def __init__(self):
        self.proc = subprocess.Popen([HELPER, "--prefix", PREFIX], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)

    def cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline().strip()
        if not out or out.startswith("error"):
            raise AssertionError("helper: %r -> %r" % (line, out))
        return out

    def create(self, nbytes):
        index, name, token, region = self.cmd("create %d" % nbytes).split()
        return int(index), name, token, int(region)

    def kill(self):
        # SIGKILL on POSIX, TerminateProcess on Windows: no destructor runs.
        self.proc.kill()
        self.proc.wait(TIMEOUT)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(TIMEOUT)
        for f in (self.proc.stdin, self.proc.stdout):
            if not f.closed:
                f.close()


def sweep(min_age_s=0):
    out = subprocess.run([HELPER, "--prefix", PREFIX, "sweep", "--min-age-s", str(min_age_s)], capture_output=True,
                         text=True, timeout=TIMEOUT, check=True).stdout
    return set(out.split())


def client(exe, name, token, region, n, how="mmap"):
    return subprocess.run([exe, "-I", "-W", "always", CLIENT, name, token, str(region), str(n), how],
                          capture_output=True, text=True, timeout=TIMEOUT)


@unittest.skipUnless(HELPER, "set MOSS_SHM_HELPER to the mossinfer_shm_helper executable")
class ShmTest(unittest.TestCase):
    def setUp(self):
        self.helper = Helper()

    def tearDown(self):
        self.helper.close()

    def test_python_clients_attach_without_tracking_or_unlinking(self):
        # Python 3.10 and 3.13: the region is read and written both ways, nothing is printed (no resource_tracker
        # warning, here or at the interpreter's exit), and the region outlives the client.
        for exe in pythons():
            major, minor = python_version(exe)
            print("  client: Python %d.%d (%s)" % (major, minor, exe), file=sys.stderr)
            for how in ["mmap"] + (["shared_memory"] if POSIX and (major, minor) >= (3, 13) else []):
                with self.subTest(python="%d.%d" % (major, minor), how=how):
                    index, name, token, region = self.helper.create(MIB)
                    self.assertEqual(self.helper.cmd("pattern %d %d" % (index, MIB)), "ok")
                    r = client(exe, name, token, region, MIB, how)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertEqual(r.stderr, "")
                    self.assertEqual(r.stdout, "")
                    if POSIX:
                        self.assertTrue(os.path.exists(shm_path(name)), "the client unlinked the region")
                    self.assertEqual(self.helper.cmd("verify %d %d" % (index, MIB)), "ok")
                    # The server's unlink after the client attached: the client's writes stay visible to it.
                    self.assertEqual(self.helper.cmd("unlink %d" % index), "ok")
                    if POSIX:
                        self.assertFalse(os.path.exists(shm_path(name)))
                    self.assertEqual(self.helper.cmd("verify %d %d" % (index, MIB)), "ok")
                    self.assertEqual(self.helper.cmd("free %d" % index), "ok")

    def test_client_refuses_a_wrong_token(self):
        index, name, token, region = self.helper.create(4096)
        wrong = ("0" if token[0] != "0" else "1") + token[1:]
        r = client(sys.executable, name, wrong, region, 16)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("another token", r.stderr)

    def test_regions_are_user_only_and_exclusive(self):
        index, name, token, region = self.helper.create(3 * MIB + 5)
        self.assertGreaterEqual(region, 4096 + 3 * MIB + 5)
        self.assertEqual(region % 4096, 0)
        if POSIX:
            st = os.stat(shm_path(name))
            self.assertEqual(st.st_mode & 0o777, 0o600)
            self.assertEqual(st.st_uid, os.geteuid())
            self.assertEqual(st.st_size, region)
        _, other, _, _ = self.helper.create(4096)
        self.assertNotEqual(name, other)
        pid = self.helper.cmd("pid")
        self.assertTrue(name.lstrip("/").replace("Local\\", "").startswith(PREFIX + pid + "-"), name)

    @unittest.skipUnless(POSIX, "named regions outlive their process on Linux only")
    def test_sigkill_leaves_only_unnamed_regions_for_the_next_sweep(self):
        # A run region the client already named (unlinked at its first step) and a scratch region never named, then
        # SIGKILL: the first is gone at once, the second until the next server's sweep at its start.
        named = self.helper.create(MIB)
        unnamed = self.helper.create(MIB)
        self.assertEqual(self.helper.cmd("unlink %d" % named[0]), "ok")
        self.helper.kill()
        self.assertFalse(os.path.exists(shm_path(named[1])))
        self.assertTrue(os.path.exists(shm_path(unnamed[1])))
        # A server's own sweep (its 10-minute floor for unproven files) removes it at once: its header proves the dead
        # pid and this PID namespace.
        self.assertIn(unnamed[1], sweep(600))
        self.assertFalse(os.path.exists(shm_path(unnamed[1])))
        self.assertNotIn(unnamed[1], sweep())

    @unittest.skipUnless(POSIX, "named regions outlive their process on Linux only")
    def test_sigkill_with_a_client_attached(self):
        # The client keeps its mapping (and the data) after the server died; the sweep removes the name only.
        index, name, token, region = self.helper.create(MIB)
        self.assertEqual(self.helper.cmd("pattern %d %d" % (index, 4096)), "ok")
        import mmap
        fd = os.open(shm_path(name), os.O_RDWR)
        buf = mmap.mmap(fd, 0)
        os.close(fd)
        self.helper.kill()
        self.assertIn(name, sweep())
        self.assertEqual(buf[4096 + 1], (1 * 7 + 3) % 256)
        buf.close()

    @unittest.skipUnless(POSIX, "the sweep is Linux only")
    def test_sweep_keeps_a_live_servers_regions(self):
        _, name, _, _ = self.helper.create(4096)
        self.assertNotIn(name, sweep())
        self.assertTrue(os.path.exists(shm_path(name)))

    @unittest.skipUnless(os.name == "nt", "Windows mappings")
    def test_mapping_vanishes_with_its_process(self):
        index, name, token, region = self.helper.create(4096)
        self.assertEqual(self.helper.cmd("pattern %d %d" % (index, 64)), "ok")
        self.assertEqual(client(sys.executable, name, token, region, 64).returncode, 0)
        self.helper.kill()
        time.sleep(0.5)
        # The tagname now creates a new, empty mapping: the client's token check refuses it.
        r = client(sys.executable, name, token, region, 64)
        self.assertEqual(r.returncode, 3, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
