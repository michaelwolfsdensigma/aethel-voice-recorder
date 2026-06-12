# Æthel Voice Recorder

A terminal voice recorder with built-in transcription and optional voice-energy analysis.
Brass instrument panel, verdigris LEDs, an engraved time counter, and an analogue-feel
VU meter — all in the terminal.

---

## What's included

| Component | What it does |
|---|---|
| **Recorder TUI** | Record to FLAC / WAV / OGG / MP3, live VU meter and waveform |
| **Transcriber** | Whisper-powered transcript → drafted reflection note |
| **Voice analysis** | Prosody + emotion arc: pitch, pace, energy, valence *(full install only)* |

---

## Requirements

- **Python 3.10+**
- **Linux or macOS** (Windows via WSL may work but is untested)
- **ffmpeg** — for MP3 encoding (`sudo apt install ffmpeg` / `brew install ffmpeg`)
- **PortAudio** — for audio capture (`sudo apt install libportaudio2 portaudio19-dev`)
- **Ollama** — for the drafting step in `aethel-transcribe` ([ollama.com](https://ollama.com))
  - Pull a model: `ollama pull qwen2.5:7b` (or any model you already have)
  - The `--no-draft` flag skips this step entirely if you just want the transcript

---

## Install

Clone the repo and run the installer:

```bash
git clone https://github.com/your-org/aethel-voice-recorder.git
cd aethel-voice-recorder
chmod +x install.sh
./install.sh
```

### Lite install (recorder + transcription, no analysis)

For machines without a dedicated GPU, or if you just want to record and transcribe:

```bash
./install.sh --lite
```

Lite mode has no torch dependency — much faster to install and lower on RAM.

### CPU-only full install

Full install but force CPU PyTorch (slower analysis, no NVIDIA GPU required):

```bash
./install.sh --cpu
```

---

## Recording

```bash
./aethel-recorder
```

| Key | Action |
|-----|--------|
| `Space` | Start recording / Pause–Resume |
| `S` | Stop and save |
| `X` | Discard current take |
| `D` | Cycle input device |
| `F` | Cycle format (FLAC / WAV / OGG / MP3) |
| `M` | Toggle mono / stereo |
| `O` | Open recordings folder |
| `Q` | Quit |

Recordings save to `~/Recordings/voice_YYYYMMDD_HHMMSS.<ext>`.

After each save, the full install automatically reads the prosody and shows a
one-line energy verdict in a notification ("Animated — high pace, bright valence").

---

## Transcription

```bash
./aethel-transcribe <recording.flac>
```

This runs three stages locally — nothing leaves your machine:

1. **Transcribe** — Whisper (small model by default) produces a timestamped transcript
2. **Read energy** — prosody analysis maps voice arc to words *(full install only)*
3. **Draft** — a local Ollama model writes a short reflection note in your voice

Output: `<recording>.reflection.md` beside the recording.

### Options

```
--whisper small|medium|large-v3   Whisper model size (default: small)
--model qwen2.5:7b                Ollama model to use for drafting
--speaker "Your Name"             Attribution in the output file
--context "After the club meetup" Background context for the drafter
--no-draft                        Transcript + energy only, skip the LLM step
```

### Example

```bash
./aethel-transcribe ~/Recordings/voice_20260612_143022.flac \
    --speaker "Sam" \
    --context "After the team retro" \
    --model qwen2.5:7b
```

---

## Install modes at a glance

| Feature | Lite | Full |
|---|---|---|
| Recorder TUI | ✓ | ✓ |
| Whisper transcription | ✓ | ✓ |
| Drafted reflection note | ✓ (needs Ollama) | ✓ (needs Ollama) |
| Prosody / pitch / pace | — | ✓ |
| Emotion arc (arousal/valence) | — | ✓ |
| Energy verdict in TUI | — | ✓ |
| Approx. extra disk (torch + models) | — | ~3–5 GB |
| GPU required | No | No (CPU works, slower) |

---

## Folder layout

```
aethel-voice-recorder/
├── app.py                      TUI
├── recorder.py                 Audio engine
├── digits.py                   Seven-segment clock
├── aethel.tcss                 Æthel palette stylesheet
├── reflect.py                  Transcription + draft tool
├── paralinguistics/
│   └── prosody_probe.py        Prosody + emotion analysis
├── aethel-recorder             Launcher → TUI
├── aethel-transcribe           Launcher → transcriber
├── install.sh                  Installer (--lite / --cpu flags)
├── requirements-lite.txt
└── requirements-full.txt
```

---

## Licence

AGPL-3.0
