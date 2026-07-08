#!/usr/bin/env python3
"""Stand-in for the Octane client: speaks the octane_bridge wire protocol so the
bridge can be validated end-to-end on the Spark without an SGI in the loop.

  python3 tools/octane_test_client.py "a red sports car" --res 512 --steps 4 \
          --host 127.0.0.1 --port 1974 --out /tmp/octane_out.png

Sends the two-line ASCII request, reads the framed [magic|status|w|h|RGB]
response (all int32 network byte order), and saves the raw RGB as a PNG.
"""
import argparse, socket, struct, sys

def recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise IOError(f"connection closed with {len(buf)}/{n} bytes")
        buf += chunk
    return buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--model", default="-", help="checkpoint name; '-' = bridge default")
    ap.add_argument("--list", action="store_true", help="print available models and exit")
    ap.add_argument("--out", default="/tmp/octane_out.png")
    a = ap.parse_args()

    if a.list:
        s = socket.create_connection((a.host, a.port), timeout=10)
        s.sendall(b"LIST\n")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        s.close()
        print(data.decode(errors="replace").strip())
        return

    s = socket.create_connection((a.host, a.port), timeout=600)
    s.settimeout(600)
    req = f"GEN {a.res} {a.steps} {a.seed} {a.model}\n{a.prompt}\n".encode()
    s.sendall(req)

    magic = recvn(s, 4)
    if magic != b"F2K1":
        print(f"bad magic: {magic!r}", file=sys.stderr); sys.exit(1)
    (status,) = struct.unpack("!i", recvn(s, 4))
    if status != 0:
        (mlen,) = struct.unpack("!i", recvn(s, 4))
        print("bridge error:", recvn(s, mlen).decode(errors="replace"), file=sys.stderr)
        sys.exit(1)
    w, h, seed = struct.unpack("!iii", recvn(s, 12))
    rgb = recvn(s, w * h * 3)
    s.close()
    print(f"got {w}x{h}, seed={seed & 0xffffffff}, {len(rgb)} bytes")

    from PIL import Image
    Image.frombytes("RGB", (w, h), rgb).save(a.out)
    print("saved", a.out)

if __name__ == "__main__":
    main()
