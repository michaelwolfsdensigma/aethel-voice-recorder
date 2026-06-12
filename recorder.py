"""
Audio engine for the Æthel Voice Recorder.

Wraps a sounddevice InputStream. The PortAudio callback runs on its own
thread; it appends captured frames and updates live level meters. The UI
thread reads `level` / `rms` / `duration` and never blocks the audio path.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


# Container formats we offer. soundfile (libsndfile) handles WAV/FLAC/OGG
# natively; MP3 is transcoded through ffmpeg if it is on the PATH.
@dataclass(frozen=True)
class AudioFormat:
    key: str
    label: str
    suffix: str
    subtype: str | None  # soundfile subtype, or None when via ffmpeg
    via_ffmpeg: bool = False


FORMATS: list[AudioFormat] = [
    AudioFormat("flac", "FLAC  · lossless, compact", ".flac", "PCM_16"),
    AudioFormat("wav", "WAV   · lossless, raw", ".wav", "PCM_16"),
    AudioFormat("ogg", "OGG   · lossy, small", ".ogg", "VORBIS"),
    AudioFormat("mp3", "MP3   · lossy, universal", ".mp3", None, via_ffmpeg=True),
]


def format_by_key(key: str) -> AudioFormat:
    for f in FORMATS:
        if f.key == key:
            return f
    return FORMATS[0]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def list_input_devices() -> list[tuple[int, str, int]]:
    """Return (index, name, max_input_channels) for devices that can capture."""
    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append((idx, dev["name"], dev["max_input_channels"]))
    return out


def default_input_index() -> int | None:
    try:
        dev = sd.query_devices(kind="input")
        return dev["index"]
    except Exception:
        return None


class Recorder:
    """A start/pause/resume/stop audio capture session."""

    def __init__(
        self,
        samplerate: int = 48000,
        channels: int = 1,
        device: int | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.device = device

        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._frames_written = 0  # only counts un-paused frames
        self._lock = threading.Lock()

        self.recording = False
        self.paused = False

        # Live meter state (read by UI thread). peak/rms in 0..1.
        self.level = 0.0  # fast peak with decay
        self.rms = 0.0
        self.peak_hold = 0.0
        self._peak_hold_t = 0.0

        # A rolling window of recent levels for the waveform history.
        self.history: queue.deque = queue.deque(maxlen=240)

    # -- audio callback -----------------------------------------------------
    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if self.paused or not self.recording:
            return
        block = indata.copy()
        with self._lock:
            self._chunks.append(block)
            self._frames_written += frames

        peak = float(np.max(np.abs(block))) if block.size else 0.0
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2))) if block.size else 0.0

        # Smooth decay on the meter so it reads like analogue VU.
        self.level = max(peak, self.level * 0.82)
        self.rms = rms
        now = time.monotonic()
        if peak >= self.peak_hold or now - self._peak_hold_t > 1.2:
            self.peak_hold = peak
            self._peak_hold_t = now
        self.history.append(rms)

    # -- transport ----------------------------------------------------------
    def start(self) -> None:
        if self.recording:
            return
        with self._lock:
            self._chunks = []
            self._frames_written = 0
        self.recording = True
        self.paused = False
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def pause(self) -> None:
        if self.recording and not self.paused:
            self.paused = True
            self.level = 0.0

    def resume(self) -> None:
        if self.recording and self.paused:
            self.paused = False

    def toggle_pause(self) -> None:
        if self.paused:
            self.resume()
        else:
            self.pause()

    def stop(self) -> np.ndarray:
        """Stop the stream and return the captured audio (frames, channels)."""
        self.recording = False
        self.paused = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.level = 0.0
        self.peak_hold = 0.0
        with self._lock:
            if self._chunks:
                return np.concatenate(self._chunks, axis=0)
            return np.zeros((0, self.channels), dtype="float32")

    # -- info ---------------------------------------------------------------
    @property
    def duration(self) -> float:
        with self._lock:
            return self._frames_written / float(self.samplerate)

    def has_audio(self) -> bool:
        with self._lock:
            return self._frames_written > 0


def save(data: np.ndarray, path: Path, fmt: AudioFormat, samplerate: int) -> Path:
    """Write captured audio to `path` in the requested format. Returns final path."""
    path = path.with_suffix(fmt.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt.via_ffmpeg:
        if not has_ffmpeg():
            raise RuntimeError("ffmpeg not found — cannot encode MP3")
        # Pipe raw float32 PCM straight into ffmpeg; no temp file.
        pcm = np.ascontiguousarray(data, dtype="float32")
        ch = pcm.shape[1] if pcm.ndim > 1 else 1
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(samplerate), "-ac", str(ch),
            "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-q:a", "2",
            str(path),
        ]
        proc = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
        return path

    sf.write(str(path), data, samplerate, subtype=fmt.subtype)
    return path
