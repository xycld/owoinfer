#!/usr/bin/env python3
"""A moss-server shared-memory client as an integration attaches (tools/server/shm.h), for tests/server/test_shm.py.
Standard library only; runs on Python 3.10 and later.

    shm_client.py NAME TOKEN REGION_BYTES N [mmap|shared_memory]

Attaches to the region, checks its header against TOKEN, checks the server's pattern (byte i = (i * 7 + 3) % 256) in
the first N payload bytes, writes its own (i * 13 % 256), and exits: 0 done, 3 refused (header or token), 4 the
server's pattern is wrong. Never unlinks the region and never registers it with multiprocessing's resource_tracker:
  Linux    os.open("/dev/shm/" + name[1:]) and mmap.mmap; or, on Python 3.13 and later, SharedMemory(track=False)
  Windows  mmap.mmap(-1, REGION_BYTES, tagname=name); the token check matters there, since a tagname recreated after
           its server died opens an empty mapping
"""
import mmap
import os
import struct
import sys

MAGIC = b"MOSSSHM\0"
LAYOUT_VERSION = 1
HEADER = struct.Struct("<8sII16sQQ")  # magic, version, server pid, token, region bytes, payload offset


def attach(name, token_hex, region_bytes, how="mmap"):
    """(buffer, payload offset, keepalive) of a checked region; raises ValueError when it is not the expected one."""
    keep = None
    if os.name == "nt":
        buf = mmap.mmap(-1, region_bytes, tagname=name)
    elif how == "shared_memory":
        from multiprocessing import shared_memory  # track=False: 3.13 and later only
        keep = shared_memory.SharedMemory(name=name.lstrip("/"), track=False)  # it prepends the "/" itself
        buf = keep.buf
    else:
        fd = os.open("/dev/shm/" + name.lstrip("/"), os.O_RDWR)
        try:
            buf = mmap.mmap(fd, 0)
        finally:
            os.close(fd)
    magic, version, _pid, token, size, payload = HEADER.unpack_from(buf, 0)
    if magic != MAGIC or version != LAYOUT_VERSION or size > len(buf) or payload != 4096:
        raise ValueError("%s is not a moss-server region" % name)
    if token != bytes.fromhex(token_hex):
        raise ValueError("%s has another token" % name)
    return buf, payload, keep


def main():
    name, token, region_bytes, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    how = sys.argv[5] if len(sys.argv) > 5 else "mmap"
    try:
        buf, off, keep = attach(name, token, region_bytes, how)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 3
    view = memoryview(buf)[off:off + n]
    if bytes(view) != bytes((i * 7 + 3) % 256 for i in range(n)):
        return 4
    view[:] = bytes(i * 13 % 256 for i in range(n))
    view.release()
    if keep is not None:
        keep.close()
    elif hasattr(buf, "close"):
        buf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
