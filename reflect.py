#!/usr/bin/env python3
"""
THE REFLECTION INSTRUMENT
=========================

Turns a voice recording into a drafted reflection note, entirely on-machine.
Three stages, all local — nothing leaves the hardware:

  1. TRANSCRIBE  — faster-whisper (GPU, cached models)
  2. READ ENERGY — prosody_probe.py (prosody + emotion arc), mapped to words
  3. DRAFT       — a local Ollama model

It exists to close the gap between talking/reflecting aloud and written output:
speak for a few minutes after a session or a build, and get back a concrete,
no-hype draft in your own register, annotated with where your voice lit up.

Requires: faster-whisper, torch, transformers, librosa, scipy (see
requirements.txt) and a running Ollama (https://ollama.com) with the drafting
model pulled, e.g. `ollama pull qwen2.5:7b`.

  python3 reflect.py <audio> [--context "..."] [--model qwen2.5:7b]
      [--speaker "Sam"] [--whisper small] [--no-draft]

Output: <audio>.reflection.md  beside the recording.

Licence: AGPL-3.0.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import re
import subprocess
import sys
import urllib.request

# --- locate the prosody/emotion engine --------------------------------------
# prosody_probe.py lives alongside this file (standalone) or in a sibling
# paralinguistics/ package (when vendored into a larger tree).
HERE = os.path.dirname(os.path.abspath(__file__))
_local = os.path.join(HERE, "prosody_probe.py")
_sibling = os.path.join(os.path.dirname(HERE), "paralinguistics", "prosody_probe.py")
PROSODY_SCRIPT = _local if os.path.isfile(_local) else _sibling

WHISPER_DEFAULT = "small"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_DEFAULT = os.environ.get("REFLECT_MODEL", "qwen2.5:7b")
WINDOW_S = 30.0

# --- proper-noun fixup ------------------------------------------------------
# Whisper reliably mangles personal/local/coined names (people, places, project
# names). Map each canonical form to the misheard variants the model actually
# produces; they are matched case-insensitively on word boundaries and replaced
# with the canonical casing. The default table is empty — populate it for your
# own vocabulary by dropping a JSON {"Canonical": ["variant", ...]} into
# ~/.config/reflect/vocab.json, or point REFLECT_VOCAB at a file.
CORRECTIONS: dict[str, list[str]] = {}


def _build_corrections():
    table = {k: list(v) for k, v in CORRECTIONS.items()}
    for p in (os.environ.get("REFLECT_VOCAB"),
              os.path.expanduser("~/.config/reflect/vocab.json")):
        if p and os.path.isfile(p):
            try:
                for canon, variants in json.load(open(p)).items():
                    table.setdefault(canon, []).extend(variants)
            except Exception as e:
                print(f"[!] vocab file {p} ignored ({e})")
    pairs = []
    for canon, variants in table.items():
        for v in variants:
            pairs.append((re.compile(rf"\b{v}\b", re.IGNORECASE), canon))
    return pairs


def fixup_proper_nouns(text, pairs):
    n = 0
    for rx, canon in pairs:
        text, k = rx.subn(canon, text)
        n += k
    return text, n


# ---------------------------------------------------------------- 1. transcribe
def transcribe(path, model_size):
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(path, vad_filter=True, beam_size=5)
    pairs = _build_corrections()
    segs, fixes = [], 0
    for s in segments:
        txt, k = fixup_proper_nouns(s.text.strip(), pairs)
        fixes += k
        segs.append((float(s.start), float(s.end), txt))
    text = " ".join(t for _, _, t in segs if t)
    del model                                   # free GPU before Ollama loads
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return text, segs, fixes


# ------------------------------------------------------------- 2. read energy
def energy_read(path):
    out = os.path.splitext(path)[0] + ".timeline.json"
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    # run prosody in a subprocess so its GPU memory is fully released on exit
    subprocess.run([sys.executable, PROSODY_SCRIPT, path, "--windows", str(WINDOW_S)],
                   env=env, capture_output=True, text=True, timeout=900)
    return json.load(open(out)) if os.path.isfile(out) else None


def _text_at(segs, t0, t1):
    snippet = " ".join(tx for (a, b, tx) in segs if b > t0 and a < t1 and tx)
    return (snippet[:240] + "…") if len(snippet) > 240 else snippet


def energy_highlights(timeline, segs):
    if not timeline:
        return None
    ws = [w for w in timeline["windows"] if w.get("valence") is not None]
    if not ws:
        return None
    win = timeline["window_s"]

    def at(w):
        return {"t": w["t"], "text": _text_at(segs, w["t"], w["t"] + win)}

    return {
        "brightest": {**at(max(ws, key=lambda w: w["valence"])),
                      "valence": max(w["valence"] for w in ws)},
        "most_energised": {**at(max(ws, key=lambda w: w["arousal"])),
                           "arousal": max(w["arousal"] for w in ws)},
        "flattest": {**at(min(ws, key=lambda w: w["arousal"])),
                     "arousal": min(w["arousal"] for w in ws)},
    }


# ------------------------------------------------------------------- 3. draft
PROMPT = """You are helping {speaker} turn a spoken reflection into a short written \
note in their own voice. Write the way a thoughtful person speaks to a smart friend: \
grounded, specific, unhurried, calm. NO hype, NO marketing voice, NO exclamation-mark \
energy, NO guru tone. Concrete over abstract: name the actual things done, the actual \
people, the actual decisions.{style_block}

{context_block}Below is a transcript of what they said, plus notes on where the voice \
carried the most genuine energy (good signals for what mattered most to them).

=== ENERGY SIGNALS ===
{energy_block}

=== TRANSCRIPT ===
{transcript}

=== YOUR TASK ===
Write a short reflection note (roughly 150–300 words) titled with a plain one-line \
summary. Lead with what was actually built or moved forward. Lean into what the voice \
lit up on; treat the flattest / lowest-energy material as less important and \
de-emphasise it accordingly. If something felt unresolved, you may note it honestly \
and briefly.

Then include a short section headed "Quiet but important": things said FLATLY but \
nonetheless consequential and easy to lose precisely because the voice did not mark \
them — a concrete commitment, a named person or a scheduled meeting, a decision already \
taken, or a genuinely novel idea or direction raised. Surface these plainly and briefly; \
do NOT inflate them into the emotional centre of the note. Omit the section entirely \
only if there is genuinely nothing of this kind.

End with 1–3 concrete open threads (things to do or decide next), drawn from BOTH what \
the voice was energised about and any quiet-but-important commitments above. \
Plain markdown. Write in the first person ('I')."""


def draft(transcript, highlights, context, model, speaker="the speaker", style=""):
    if highlights:
        eb = "\n".join(
            f"- {k}: \"{v['text']}\"" for k, v in {
                "lit up most (positive)": highlights["brightest"],
                "most energised": highlights["most_energised"],
                "flattest / lowest energy": highlights["flattest"],
            }.items() if v.get("text"))
    else:
        eb = "(energy read unavailable)"
    cb = f"Context they gave: {context}\n\n" if context else ""
    sb = f"\nVoice/style notes: {style}" if style else ""
    prompt = PROMPT.format(speaker=speaker, style_block=sb,
                           context_block=cb, energy_block=eb, transcript=transcript)
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.7, "num_ctx": 8192}}).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"].strip()


# -------------------------------------------------------------------- compose
def main():
    ap = argparse.ArgumentParser(description="The Reflection Instrument")
    ap.add_argument("audio")
    ap.add_argument("--context", default="", help="one line of context for the draft")
    ap.add_argument("--model", default=LLM_DEFAULT, help="Ollama model for drafting")
    ap.add_argument("--whisper", default=WHISPER_DEFAULT, help="faster-whisper size")
    ap.add_argument("--speaker", default="the speaker", help="who is reflecting")
    ap.add_argument("--style", default="", help="optional voice/style notes for the draft")
    ap.add_argument("--no-draft", action="store_true", help="transcript + energy only")
    args = ap.parse_args()

    path = os.path.abspath(args.audio)
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")

    print(f"[1/3] transcribing ({args.whisper})…")
    transcript, segs, fixes = transcribe(path, args.whisper)
    extra = f", {fixes} proper-noun fixes" if fixes else ""
    print(f"      {len(transcript)} chars, {len(segs)} segments{extra}")

    print("[2/3] reading energy…")
    timeline = energy_read(path)
    hi = energy_highlights(timeline, segs)

    note = None
    if not args.no_draft and transcript.strip():
        print(f"[3/3] drafting with {args.model}…")
        try:
            note = draft(transcript, hi, args.context, args.model,
                         speaker=args.speaker, style=args.style)
        except Exception as e:
            print(f"[!] draft failed ({e.__class__.__name__}: {e}); writing transcript + energy only.")

    # ---- assemble the markdown ----
    now = datetime.datetime.now()
    lines = [f"# Reflection — {now:%d %B %Y, %H:%M}", ""]
    if args.context:
        lines += [f"*{args.context}*", ""]
    if note:
        lines += ["## The note", "", note, "", "---", ""]
    if hi:
        lines += ["## What your voice did", ""]
        for label, key in [("Lit up most", "brightest"),
                            ("Most energised", "most_energised"),
                            ("Flattest", "flattest")]:
            h = hi[key]
            if h.get("text"):
                lines.append(f"- **{label}** (~{h['t']:.0f}s): \"{h['text']}\"")
        lines.append("")
    if timeline:
        def arc(k):
            v = [w[k] for w in timeline["windows"] if w.get(k) is not None]
            return f"mean {sum(v)/len(v):.2f}" if v else "—"
        lines += ["## Energy arc",
                  f"- arousal {arc('arousal')} · valence {arc('valence')} · "
                  f"dominance {arc('dominance')}", ""]
    lines += ["## Transcript", "", transcript, ""]

    out = os.path.splitext(path)[0] + ".reflection.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))

    if note:
        print("\n" + "=" * 60)
        print(note)
        print("=" * 60)
    print(f"\n[*] reflection written: {out}")


if __name__ == "__main__":
    main()
