#!/usr/bin/env python3
"""Cross-platform launcher for the F2K web UI (Linux + Windows).

Starts the persistent worker if it isn't already listening, waits for it to come
up, then runs the Flask front-end. Replaces the bash-only start.sh on Windows;
either works on Linux.

Run it with the venv interpreter so Flask is on the path, e.g.
  Linux:    .venv/bin/python tools/webui/start.py
  Windows:  .venv\\Scripts\\python tools\\webui\\start.py   (or just: tools\\webui\\start.bat)

Env: F2K_WEB_PASSWORD, F2K_WEB_USER, F2K_WORKER_PORT (default 8765), F2K_WEB_PORT.
"""
import os, sys, socket, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PORT = int(os.environ.get("F2K_WORKER_PORT", "8765"))


def find_serve():
    """Locate the worker binary across Linux / MSVC (Release subdir) layouts."""
    names = ["serve.exe", "serve"] if os.name == "nt" else ["serve"]
    dirs = [os.path.join(ROOT, "build"),
            os.path.join(ROOT, "build", "Release"),
            os.path.join(ROOT, "build", "Debug")]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def worker_up():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            return True
    except OSError:
        return False


def main():
    if worker_up():
        print(f"[start] worker already listening on :{PORT}")
    else:
        serve = find_serve()
        if not serve:
            print("[start] worker binary not found — build the 'serve' target first "
                  "(cmake --build build --target serve)", file=sys.stderr)
            sys.exit(1)
        print(f"[start] launching worker: {serve}  (resident model load ~15s)...")
        log = open(os.path.join(tempfile.gettempdir(), "f2k_serve.log"), "w")
        args = [serve, "--port", str(PORT)]
        # Preload the default model's encoder (web UI defaults to klein-4B).
        default_root = os.path.join(os.path.expanduser("~"), "models",
                                    os.environ.get("F2K_DEFAULT_MODEL", "flux2-klein-4B"))
        if os.path.isdir(os.path.join(default_root, "qwen3_f2k")):
            args += ["--model", default_root]
        subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        for _ in range(90):
            if worker_up():
                break
            time.sleep(1)
        print("[start] worker:", "ready" if worker_up() else "not responding yet "
              "(check the worker log)")

    os.environ.setdefault("F2K_WEB_PASSWORD", "rocket")
    port = os.environ.get("F2K_WEB_PORT", "5000")
    print(f"[start] web UI → http://localhost:{port}  (user={os.environ.get('F2K_WEB_USER','chris')})")
    # run the Flask app with this same interpreter (so its venv has Flask)
    subprocess.run([sys.executable, os.path.join(HERE, "app.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
