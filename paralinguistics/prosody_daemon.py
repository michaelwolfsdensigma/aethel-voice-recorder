#!/usr/bin/env python3
"""
Æthel Prosody Daemon — keeps the wav2vec2 emotion model warm in VRAM so
single-clip and windowed analysis runs return in seconds rather than minutes.

Usage
-----
  Start manually:   python3 prosody_daemon.py
  Via systemd:      systemctl --user start aethel-prosody
  Stop:             systemctl --user stop aethel-prosody
  Status/socket:    systemctl --user status aethel-prosody

Protocol
--------
UNIX domain socket at $XDG_RUNTIME_DIR/aethel-prosody.sock.
Newline-delimited JSON: send one request line, receive one response line.

Request:
  {
    "path":     "/abs/path/to/audio.wav",   # required
    "mode":     "single" | "windowed",      # default: "single"
    "windows":  30.0,                        # seconds per window (windowed mode)
    "no_tier2": false,
    "hpf":      60.0,
    "notch":    50.0
  }

Response (single):   aethel.prosody/0.1 dict (same as prosody_probe.py sidecar)
Response (windowed): aethel.timeline/0.1 dict (same as prosody_probe.py timeline)
Response (error):    {"error": "..."}

The client (prosody_probe.py) writes the sidecar file itself from the response,
so the daemon never touches the filesystem.
"""
import datetime
import json
import logging
import math
import os
import signal
import socket
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SOCKET_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
    "aethel-prosody.sock",
)
LOG = logging.getLogger("aethel-prosody")


# --------------------------------------------------------------------------- model load

def _warm_model():
    """Pre-load wav2vec2 into VRAM.  Called once at daemon start."""
    import numpy as np
    from prosody_probe import tier2_emotion
    LOG.info("Loading wav2vec2 emotion model (first run may download ~1.2 GB)…")
    dummy = np.zeros(16000, dtype=np.float32)
    try:
        tier2_emotion(dummy)
        LOG.info("Model warm and ready.")
    except Exception as exc:
        LOG.warning("Model pre-load failed (%s) — will retry on first request.", exc)


# --------------------------------------------------------------------------- request handler

def _handle_conn(conn: socket.socket):
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(8192)
            if not chunk:
                return
            buf += chunk
        req = json.loads(buf.split(b"\n", 1)[0].strip())
        result = _process(req)
        conn.sendall(json.dumps(result, allow_nan=False).encode() + b"\n")
    except Exception as exc:
        LOG.exception("handler error")
        try:
            conn.sendall(json.dumps({"error": str(exc)}).encode() + b"\n")
        except Exception:
            pass
    finally:
        conn.close()


def _process(req: dict) -> dict:
    import librosa
    import numpy as np
    from prosody_probe import analyze, hum_notch, high_pass, _features

    path = req.get("path", "")
    if not os.path.isfile(path):
        return {"error": f"no such file: {path}"}

    hpf     = float(req.get("hpf",   60.0))
    notch   = float(req.get("notch", 50.0))
    no_t2   = bool(req.get("no_tier2", False))
    mode    = req.get("mode", "single")
    windows = req.get("windows", 30.0)
    do_t2   = not no_t2

    y, sr = librosa.load(path, sr=None, mono=True)
    y_raw = y.copy()                     # pre-filter signal for the fan/AC test
    y = hum_notch(y, sr, notch)
    y = high_pass(y, sr, hpf)
    now = datetime.datetime.now().isoformat(timespec="seconds")

    if mode == "windowed":
        win = int(windows * sr)
        tl  = []
        n   = max(1, int(math.ceil(len(y) / win)))
        for i in range(n):
            seg = y[i * win:(i + 1) * win]
            if len(seg) < 0.5 * sr:
                continue
            seg_raw = y_raw[i * win:(i + 1) * win]
            t0 = i * windows
            r  = analyze(seg, sr, do_t2, y_raw=seg_raw)
            f  = _features(r)
            f["t"] = round(t0, 1)
            tl.append(f)
        return {
            "schema":       "aethel.timeline/0.1",
            "source":       path,
            "generated":    now,
            "window_s":     windows,
            "high_pass_hz": hpf,
            "notch_hz":     notch,
            "windows":      tl,
        }

    # single-clip mode
    res = analyze(y, sr, do_t2, y_raw=y_raw)
    return {
        "schema":       "aethel.prosody/0.1",
        "source":       path,
        "generated":    now,
        "high_pass_hz": hpf,
        "notch_hz":     notch,
        **res,
    }


# --------------------------------------------------------------------------- server loop

def serve():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    srv.listen(4)
    LOG.info("Listening at %s", SOCKET_PATH)

    _warm_model()

    def _shutdown(sig, _frame):
        LOG.info("Signal %s — shutting down.", sig)
        srv.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()


# --------------------------------------------------------------------------- entry point

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    serve()
