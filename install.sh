#!/usr/bin/env bash
# Æthel Voice Recorder — installer
#
# Usage:
#   ./install.sh           full install (recorder + transcription + analysis)
#   ./install.sh --lite    lite install (recorder + transcription only, no GPU analysis)
#   ./install.sh --cpu     full install but force CPU-only PyTorch (saves ~2 GB)
#
# After install:
#   ./aethel-recorder      launch the TUI
#   ./aethel-transcribe <recording.flac> [--model qwen2.5:7b] [--speaker "You"]

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# ── parse flags ─────────────────────────────────────────────────────────────
LITE=false
FORCE_CPU=false
for arg in "$@"; do
    case "$arg" in
        --lite)      LITE=true ;;
        --cpu)       FORCE_CPU=true ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $arg  (try --help)"
            exit 1 ;;
    esac
done

# ── banner ───────────────────────────────────────────────────────────────────
echo ""
echo "  ═══❖ ÆTHEL VOICE RECORDER — Installer ❖═══"
if $LITE; then
    echo "  Mode: LITE  (recorder + transcription)"
else
    echo "  Mode: FULL  (recorder + transcription + voice analysis)"
fi
echo ""

# ── system checks ────────────────────────────────────────────────────────────
echo "[ ] Checking system requirements…"

# Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo "  ✗ python3 not found. Install Python 3.10+ and re-run."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_OK=$(python3 -c "import sys; print('ok' if sys.version_info >= (3, 10) else 'old')")
if [[ "$PY_OK" != "ok" ]]; then
    echo "  ✗ Python $PY_VER found — need 3.10+."
    exit 1
fi
echo "  ✓ Python $PY_VER"

# ffmpeg (needed for MP3 encoding; optional — recorder works without it)
if command -v ffmpeg &>/dev/null; then
    echo "  ✓ ffmpeg found (MP3 encoding available)"
else
    echo "  ⚠ ffmpeg not found — MP3 format will be unavailable."
    echo "    Install with: sudo apt install ffmpeg   (or brew install ffmpeg on Mac)"
fi

# PortAudio (needed by sounddevice)
if pkg-config --exists portaudio-2.0 2>/dev/null || ldconfig -p 2>/dev/null | grep -q portaudio; then
    echo "  ✓ PortAudio found"
else
    echo "  ⚠ PortAudio may be missing. If recording fails, install with:"
    echo "    sudo apt install libportaudio2 portaudio19-dev"
fi

echo ""

# ── venv ─────────────────────────────────────────────────────────────────────
echo "[ ] Creating virtual environment…"
python3 -m venv .venv
echo "  ✓ .venv created"
echo ""

PIP=".venv/bin/pip"
PYTHON=".venv/bin/python3"

"$PIP" install --quiet --upgrade pip

# ── lite deps ────────────────────────────────────────────────────────────────
echo "[ ] Installing recorder + transcription dependencies…"
"$PIP" install --quiet -r requirements-lite.txt
echo "  ✓ textual, sounddevice, soundfile, numpy, faster-whisper"
echo ""

# ── full deps ────────────────────────────────────────────────────────────────
if ! $LITE; then
    echo "[ ] Installing voice analysis dependencies…"

    # Detect GPU unless --cpu forced
    GPU_AVAILABLE=false
    if ! $FORCE_CPU && command -v nvidia-smi &>/dev/null; then
        GPU_AVAILABLE=true
    fi

    if $GPU_AVAILABLE; then
        echo "  · GPU detected — installing CUDA-enabled PyTorch"
        "$PIP" install --quiet torch torchaudio \
            --index-url https://download.pytorch.org/whl/cu121
    else
        echo "  · Installing CPU-only PyTorch (pass --cpu to force this explicitly)"
        "$PIP" install --quiet torch torchaudio \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    "$PIP" install --quiet -r requirements-full.txt
    echo "  ✓ torch, torchaudio, transformers, librosa, scipy"

    # Mark this as a full install so the TUI activates the analysis sidecar
    touch .analysis
    echo ""
    echo "  Note: the first recording after launch will load the emotion model"
    echo "  (~1.2 GB, cached after first download). Subsequent loads are fast."
    echo ""
fi

# ── Whisper model pre-download ───────────────────────────────────────────────
echo "[ ] Pre-downloading Whisper model (small, ~240 MB)…"
echo "    (skipping if already cached)"
"$PYTHON" - <<'PY' || echo "  ⚠ Whisper pre-download skipped (will download on first use)"
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8")
print("  ✓ Whisper small model ready")
PY
echo ""

# ── launchers ────────────────────────────────────────────────────────────────
echo "[ ] Setting up launchers…"
chmod +x aethel-recorder aethel-transcribe
echo "  ✓ ./aethel-recorder      — launch the TUI"
echo "  ✓ ./aethel-transcribe    — transcribe a recording"
echo ""

# ── done ─────────────────────────────────────────────────────────────────────
echo "  ═══❖ Installation complete ❖═══"
echo ""
echo "  Start recording:    ./aethel-recorder"
echo ""
if ! $LITE; then
    echo "  Transcribe a clip:  ./aethel-transcribe <recording.flac>"
    echo "                      ./aethel-transcribe <recording.flac> --no-draft"
    echo "                      (requires Ollama running with a model pulled)"
    echo ""
    echo "  Voice analysis runs automatically after each save in the TUI."
fi
if $LITE; then
    echo "  Transcribe a clip:  ./aethel-transcribe <recording.flac> --no-draft"
    echo "                      (transcript only, no LLM draft)"
    echo ""
    echo "  To upgrade to the full install later: ./install.sh"
fi
echo ""
