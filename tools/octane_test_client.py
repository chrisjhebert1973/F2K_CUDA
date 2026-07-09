#!/usr/bin/env python3
"""Stand-in for the Octane client: speaks the octane_bridge wire protocol so the
bridge can be validated end-to-end on the Spark without an SGI in the loop.

  python3 tools/octane_test_client.py "a red sports car" --count 4 --res 512 \
          --steps 4 --model flux2-klein-4B --out /tmp/octane_out

Sends "GEN <res> <steps> <seed> <count> <model>\\n<prompt>\\n", then reads the
tagged message stream (PROGRESS / IMAGE / DONE / ERROR; all int32 network byte
order), printing progress and saving each IMAGE as <out>_<idx>.png.
"""
import argparse, socket, struct, sys

MAGIC = b"F2K1"
MSG_PROGRESS, MSG_IMAGE, MSG_DONE, MSG_ERROR = 1, 2, 3, 4

def recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise IOError(f"connection closed with {len(buf)}/{n} bytes")
        buf += chunk
    return buf

def u32(s):
    return struct.unpack("!I", recvn(s, 4))[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--model", default="-", help="checkpoint name; '-' = bridge default")
    ap.add_argument("--list", action="store_true", help="print available models and exit")
    ap.add_argument("--remix", default="", help="init image path (img2img)")
    ap.add_argument("--strength", type=float, default=0.6, help="remix strength 0.05..1.0")
    ap.add_argument("--out", default="/tmp/octane_out", help="output base path (_<idx>.png)")
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

    from PIL import Image
    s = socket.create_connection((a.host, a.port), timeout=600)
    s.settimeout(600)
    if a.remix:
        # center-crop to square, cap at 1024, send raw RGB after the header
        im = Image.open(a.remix).convert("RGB")
        w, h = im.size
        sq = min(w, h)
        im = im.crop(((w - sq) // 2, (h - sq) // 2, (w - sq) // 2 + sq, (h - sq) // 2 + sq))
        if sq > 1024:
            im = im.resize((1024, 1024)); sq = 1024
        rgb = im.tobytes()
        st = max(5, min(100, int(a.strength * 100)))
        hdr = f"REMIX {a.res} {a.steps} {a.seed} {a.count} {st} {sq} {sq} {a.model}\n{a.prompt}\n"
        s.sendall(hdr.encode())
        s.sendall(rgb)
    else:
        req = f"GEN {a.res} {a.steps} {a.seed} {a.count} {a.model}\n{a.prompt}\n".encode()
        s.sendall(req)

    saved = 0
    while True:
        if recvn(s, 4) != MAGIC:
            print("bad magic", file=sys.stderr); sys.exit(1)
        mtype = u32(s)
        if mtype == MSG_PROGRESS:
            idx, total, permille = u32(s), u32(s), u32(s)
            phase = recvn(s, u32(s)).decode(errors="replace")
            print(f"\r  [{permille/10:5.1f}%] image {idx+1}/{total}  {phase:<10}", end="", flush=True)
        elif mtype == MSG_IMAGE:
            idx, w, h, seed = u32(s), u32(s), u32(s), u32(s)
            rgb = recvn(s, w * h * 3)
            fn = f"{a.out}_{idx}.png"
            Image.frombytes("RGB", (w, h), rgb).save(fn)
            saved += 1
            print(f"\r  image {idx} -> {fn}  {w}x{h} seed={seed}" + " " * 12)
        elif mtype == MSG_DONE:
            count = u32(s)
            print(f"DONE: {count} image(s), {saved} saved")
            break
        elif mtype == MSG_ERROR:
            print("\nbridge error:", recvn(s, u32(s)).decode(errors="replace"), file=sys.stderr)
            sys.exit(1)
        else:
            print(f"\nunknown message type {mtype}", file=sys.stderr); sys.exit(1)
    s.close()

if __name__ == "__main__":
    main()
