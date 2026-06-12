#!/usr/bin/env python3
"""
ÆTHEL · VOICE RECORDER

A terminal voice recorder in the Greco-Roman steampunk house style.
Brass instrument panel, verdigris LEDs, an engraved seven-segment counter,
and a VU meter that reads like analogue.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.widgets import Static

import digits
import recorder as rec

# -- Æthel palette ----------------------------------------------------------
INK = "#0a0602"
PARCHMENT = "#f5e6c8"
PARCH_LIGHT = "#fff5e1"
BRASS = "#b8860b"
BRASS_LIGHT = "#daaf3c"
BRASS_DARK = "#8b6914"
BRONZE = "#cd7f32"
BRONZE_DARK = "#a05f1e"
VERDIGRIS = "#2e7d6e"
VERD_LIGHT = "#46a08c"
VERD_DARK = "#1a4d43"
HOT = "#d65a3a"  # over-level warning, warm not garish

REC_DIR = Path.home() / "Recordings"

# -- Paralinguistic sidecar (prosody + emotion) -----------------------------
# Runs prosody_probe.py under the package venv. Only active when the full
# install was used (install.sh without --lite). A .analysis marker file is
# created by the full installer; its absence silently disables the sidecar.
_PKG_ROOT = Path(__file__).resolve().parent
PROSODY_PY = _PKG_ROOT / ".venv" / "bin" / "python3"
PROSODY_SCRIPT = _PKG_ROOT / "paralinguistics" / "prosody_probe.py"
PROSODY_MIN_SECONDS = 1.5  # don't bother analysing very short blips
_ANALYSIS_ENABLED = (_PKG_ROOT / ".analysis").exists()


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


class StatusPanel(Static):
    """LED + transport state on the left, device/format on the right."""

    def render_state(self, *, recording: bool, paused: bool, blink: bool,
                     device_name: str, fmt_label: str, channels: int) -> None:
        if not recording:
            led_color, led, state = VERD_DARK, "●", "READY"
            state_color = PARCHMENT
        elif paused:
            led_color, led, state = BRASS_LIGHT, "❚❚", "PAUSED"
            state_color = BRASS_LIGHT
        else:
            led_color = HOT if blink else BRONZE_DARK
            led, state = "●", "RECORDING"
            state_color = HOT

        t = Text()
        t.append(f"  {led} ", style=f"bold {led_color}")
        t.append(f"{state:<10}", style=f"bold {state_color}")
        ch = "mono" if channels == 1 else "stereo"
        right = f"{device_name}  ·  {fmt_label}  ·  {ch}  "
        pad = max(1, self.size.width - len(state) - len(right) - 6)
        t.append(" " * pad)
        t.append(right, style=BRASS)
        self.update(t)


class Clock(Static):
    """Big engraved seven-segment time counter."""

    def render_time(self, seconds: float, *, active: bool) -> None:
        color = BRASS_LIGHT if active else BRASS_DARK
        art = digits.big(digits.format_time(seconds))
        self.update(Text(art, style=f"bold {color}"))


class Meter(Static):
    """Horizontal VU meter: verdigris → brass → hot, with peak hold."""

    def render_level(self, level: float, peak: float) -> None:
        width = max(10, self.size.width - 8)
        filled = int(round(min(level, 1.0) * width))
        peak_cell = int(round(min(peak, 1.0) * (width - 1)))

        t = Text()
        t.append("  ", style=INK)
        for i in range(width):
            p = i / width
            if p < 0.6:
                on, off = VERD_LIGHT, VERD_DARK
            elif p < 0.85:
                on, off = BRASS_LIGHT, BRASS_DARK
            else:
                on, off = HOT, BRONZE_DARK
            if i == peak_cell and peak > 0.02:
                t.append("┃", style=f"bold {PARCH_LIGHT}")
            elif i < filled:
                t.append("█", style=on)
            else:
                t.append("─", style=off)
        db = 20 * np.log10(peak) if peak > 1e-4 else -60.0
        t.append(f"  {db:5.1f} dB", style=BRASS)
        self.update(t)


class Waveform(Static):
    """Scrolling sparkline of recent RMS levels."""

    _BARS = " ▁▂▃▄▅▆▇█"

    def render_wave(self, history) -> None:
        width = max(10, self.size.width - 4)
        vals = list(history)[-width:]
        if not vals:
            vals = [0.0]
        t = Text("  ", style=INK)
        for v in vals:
            idx = min(len(self._BARS) - 1, int(v * 1.8 * (len(self._BARS) - 1)))
            if idx <= 2:
                color = VERD_DARK
            elif idx <= 5:
                color = VERDIGRIS
            else:
                color = BRASS
            t.append(self._BARS[idx], style=color)
        self.update(t)


class SessionLog(Static):
    """Recordings saved this session."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.entries: list[str] = []

    def add(self, path: Path, duration: float, size: int) -> None:
        ts = datetime.now().strftime("%H:%M")
        dur = digits.format_time(duration)
        self.entries.insert(0, f"{ts}  {path.name:<30} {dur:>8}  {_human_size(size):>8}")
        self._refresh()

    def _refresh(self) -> None:
        t = Text()
        t.append("  ❧ SESSION LOG\n", style=f"bold {BRONZE}")
        if not self.entries:
            t.append("    no recordings yet — press ", style=VERD_DARK)
            t.append("Space", style=BRASS)
            t.append(" to begin\n", style=VERD_DARK)
        for line in self.entries[:8]:
            t.append(f"    {line}\n", style=PARCHMENT)
        self.update(t)

    def on_mount(self) -> None:
        self._refresh()


class RecorderApp(App):
    CSS_PATH = "aethel.tcss"
    TITLE = "ÆTHEL · VOICE RECORDER"

    BINDINGS = [
        ("space", "transport", "Rec/Pause"),
        ("s", "stop_save", "Stop+Save"),
        ("x", "discard", "Discard"),
        ("d", "cycle_device", "Device"),
        ("f", "cycle_format", "Format"),
        ("m", "toggle_channels", "Mono/Stereo"),
        ("o", "open_folder", "Folder"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.devices = rec.list_input_devices()
        default_idx = rec.default_input_index()
        self.dev_pos = 0
        for i, (idx, _n, _c) in enumerate(self.devices):
            if idx == default_idx:
                self.dev_pos = i
                break
        self.formats = [f for f in rec.FORMATS if not (f.via_ffmpeg and not rec.has_ffmpeg())]
        self.fmt_pos = 0
        self.channels = 1
        self.recorder: rec.Recorder | None = None
        self._last_duration = 0.0

    # -- layout -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static(self._banner(), id="banner")
            yield StatusPanel(id="status")
            with Center(id="clockwrap"):
                yield Clock(id="clock")
            with Vertical(id="meters"):
                yield Static("  ⟁ INPUT LEVEL", id="meterlabel")
                yield Meter(id="meter")
                yield Waveform(id="wave")
            yield SessionLog(id="log")
            yield Static(self._keyhints(), id="hints")

    def _banner(self) -> Text:
        line = "═" * 3
        t = Text()
        t.append(f"  {line}❖ ", style=BRONZE)
        t.append("ÆTHEL VOICE RECORDER", style=f"bold {BRASS_LIGHT}")
        t.append(f" ❖{line}", style=BRONZE)
        return t

    def _keyhints(self) -> Text:
        pairs = [
            ("Space", "Rec/Pause"), ("S", "Stop+Save"), ("X", "Discard"),
            ("D", "Device"), ("F", "Format"), ("M", "Ch"), ("O", "Folder"), ("Q", "Quit"),
        ]
        t = Text("  ")
        for k, v in pairs:
            t.append(f" {k} ", style=f"bold {INK} on {BRASS}")
            t.append(f" {v}  ", style=PARCHMENT)
        return t

    # -- lifecycle ----------------------------------------------------------
    def on_mount(self) -> None:
        self.screen.styles.background = INK
        self.set_interval(1 / 24, self.tick)
        self.refresh_status()
        self.query_one(Clock).render_time(0.0, active=False)
        self.query_one(Meter).render_level(0.0, 0.0)
        self.query_one(Waveform).render_wave([])

    def tick(self) -> None:
        r = self.recorder
        active = bool(r and r.recording and not r.paused)
        if r and r.recording:
            self.query_one(Clock).render_time(r.duration, active=active)
            self.query_one(Meter).render_level(r.level, r.peak_hold)
            self.query_one(Waveform).render_wave(r.history)
        blink = int(time.monotonic() * 2) % 2 == 0
        self.refresh_status(blink=blink)

    def refresh_status(self, blink: bool = True) -> None:
        r = self.recorder
        dev_name = self.devices[self.dev_pos][1] if self.devices else "no input"
        # Trim very long ALSA names.
        dev_name = dev_name.split(",")[0][:28]
        fmt = self.formats[self.fmt_pos]
        self.query_one(StatusPanel).render_state(
            recording=bool(r and r.recording),
            paused=bool(r and r.paused),
            blink=blink,
            device_name=dev_name,
            fmt_label=fmt.key.upper(),
            channels=self.channels,
        )

    # -- actions ------------------------------------------------------------
    def action_transport(self) -> None:
        r = self.recorder
        if r is None or not r.recording:
            self._start()
        else:
            r.toggle_pause()
            self.refresh_status()

    def _start(self) -> None:
        if not self.devices:
            self.notify("No input device available", severity="error")
            return
        dev_index = self.devices[self.dev_pos][0]
        max_ch = self.devices[self.dev_pos][2]
        ch = min(self.channels, max_ch)
        try:
            self.recorder = rec.Recorder(samplerate=48000, channels=ch, device=dev_index)
            self.recorder.start()
        except Exception as e:  # noqa: BLE001
            self.notify(f"Could not start: {e}", severity="error")
            self.recorder = None
            return
        self.notify("Recording…", timeout=2)
        self.refresh_status()

    def action_stop_save(self) -> None:
        r = self.recorder
        if r is None or not r.recording:
            return
        if not r.has_audio():
            self.action_discard()
            return
        data = r.stop()
        self._last_duration = len(data) / float(r.samplerate)
        self.recorder = None
        self._save(data, r.samplerate)
        self.query_one(Clock).render_time(self._last_duration, active=False)
        self.query_one(Meter).render_level(0.0, 0.0)
        self.refresh_status()

    @work(thread=True)
    def _save(self, data: np.ndarray, samplerate: int) -> None:
        fmt = self.formats[self.fmt_pos]
        name = "voice_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        target = REC_DIR / name
        try:
            path = rec.save(data, target, fmt, samplerate)
            size = path.stat().st_size
            dur = len(data) / float(samplerate)
            self.call_from_thread(self.query_one(SessionLog).add, path, dur, size)
            self.call_from_thread(
                lambda: self.notify(f"Saved {path.name}", title="Done")
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.call_from_thread(
                lambda: self.notify(f"Save failed: {msg}", title="Error", severity="error")
            )
            return
        # Still on the save worker thread — analyse the clip in the background.
        self._analyse_prosody(path, dur)

    def _analyse_prosody(self, path: Path, dur: float) -> None:
        """Generate the prosody+emotion sidecar and surface a one-line verdict.
        Best-effort: any failure is swallowed so it can never disrupt recording."""
        if not _ANALYSIS_ENABLED:
            return
        if dur < PROSODY_MIN_SECONDS or not PROSODY_PY.exists() or not PROSODY_SCRIPT.exists():
            return
        try:
            self.call_from_thread(lambda: self.notify("Reading the voice…", timeout=2))
            env = dict(os.environ, HF_HUB_OFFLINE="1")
            subprocess.run(
                [str(PROSODY_PY), str(PROSODY_SCRIPT), str(path)],
                capture_output=True, text=True, env=env, timeout=180,
            )
            sidecar = path.with_suffix(".prosody.json")
            if not sidecar.exists():
                return
            data = json.loads(sidecar.read_text())
            verdict = (data.get("baseline_match") or {}).get("verdict")
            if verdict:
                self.call_from_thread(
                    lambda: self.notify(verdict, title="Voice", timeout=7)
                )
        except Exception:  # noqa: BLE001
            pass  # never let analysis break the recorder

    def action_discard(self) -> None:
        r = self.recorder
        if r is None or not r.recording:
            return
        r.stop()
        self.recorder = None
        self.query_one(Clock).render_time(0.0, active=False)
        self.query_one(Meter).render_level(0.0, 0.0)
        self.query_one(Waveform).render_wave([])
        self.notify("Discarded", timeout=2)
        self.refresh_status()

    def action_cycle_device(self) -> None:
        if self.recorder and self.recorder.recording:
            self.notify("Stop recording before changing device", severity="warning")
            return
        if not self.devices:
            return
        self.dev_pos = (self.dev_pos + 1) % len(self.devices)
        self.refresh_status()

    def action_cycle_format(self) -> None:
        self.fmt_pos = (self.fmt_pos + 1) % len(self.formats)
        self.refresh_status()

    def action_toggle_channels(self) -> None:
        if self.recorder and self.recorder.recording:
            self.notify("Stop recording before changing channels", severity="warning")
            return
        self.channels = 2 if self.channels == 1 else 1
        self.refresh_status()

    def action_open_folder(self) -> None:
        REC_DIR.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.Popen(["xdg-open", str(REC_DIR)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.notify(f"Opened {REC_DIR}", timeout=2)

    def action_quit(self) -> None:
        if self.recorder and self.recorder.recording:
            self.recorder.stop()
        self.exit()


def main() -> None:
    RecorderApp().run()


if __name__ == "__main__":
    main()
