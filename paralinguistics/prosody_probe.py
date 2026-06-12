#!/usr/bin/env python3
"""
Prosody + emotion probe — prototype for the Æthel paralinguistic sidecar.

Tier 1 (prosody, deterministic, CPU): pitch contour, loudness dynamics,
        speaking rate proxy, pauses, final-contour direction.  librosa only.
Tier 2 (emotion, inferred, GPU): audeering wav2vec2 dimensional model →
        arousal / dominance / valence in ~0..1.

Run with the hermes-agent venv (has librosa + torch + transformers + CUDA):
    ~/Projects/hermes-agent/.venv/bin/python3 prosody_probe.py <audio> [--no-tier2]

Writes <audio>.prosody.json next to the file and prints a human summary.
This is a prototype: the emotion layer is a noisy, population-trained estimate,
not ground truth.  The prosody layer is the reliable part.
"""
import sys, os, json, math, argparse, datetime

import numpy as np
import librosa
import scipy.signal as sps

MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SPEECH_FMIN, SPEECH_FMAX = 65.0, 400.0   # adult speech F0 search range (Hz)
DEFAULT_BASELINE = os.path.expanduser("~/Recordings/Voice Sample Prime.baseline.json")
_EMO_CACHE = None                        # lazily-loaded (processor, model, device)
MATCH_KEYS = ["rate", "speech_ratio", "pauses_per_min",
              "arousal", "valence", "dominance", "pitch_hz"]
HPF_DEFAULT = 60.0                       # high-pass cutoff (Hz) — kills rumble but
                                         # stays below Michael's ~77 Hz fundamental


def high_pass(y, sr, cutoff=HPF_DEFAULT):
    """4th-order Butterworth high-pass to remove handling/proximity rumble."""
    if cutoff <= 0:
        return y
    sos = sps.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def hum_notch(y, sr, mains=50.0, q=30.0):
    """Notch out mains hum + harmonics (50/100/150… Hz) that the high-pass
    passes and that pyin otherwise latches onto, faking a low pitch."""
    if mains <= 0:
        return y
    f = mains
    while f < sr / 2 * 0.95:
        b, a = sps.iirnotch(f, q, fs=sr)
        y = sps.filtfilt(b, a, y)
        f += mains
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- Tier 1
def hz_to_semitones(f0, ref):
    return 12.0 * np.log2(f0 / ref)


def tier1_prosody(y, sr):
    dur = len(y) / sr

    # --- Pitch (F0) via probabilistic YIN ---
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=SPEECH_FMIN, fmax=SPEECH_FMAX, sr=sr, frame_length=2048
    )
    times = librosa.times_like(f0, sr=sr)
    voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced = voiced[~np.isnan(voiced)]

    pitch = {"detected": False}
    if voiced.size:
        median_hz = float(np.median(voiced))
        semis = hz_to_semitones(voiced, median_hz)
        # final-contour direction: slope over the last voiced run
        vt = times[voiced_flag]
        vf = f0[voiced_flag]
        good = ~np.isnan(vf)
        vt, vf = vt[good], vf[good]
        contour = "flat"
        slope_st_per_s = 0.0
        if vf.size >= 4:
            tail = max(4, vf.size // 4)            # last ~quarter, >=4 frames
            tt, tf = vt[-tail:], hz_to_semitones(vf[-tail:], median_hz)
            slope_st_per_s = float(np.polyfit(tt - tt[0], tf, 1)[0])
            if slope_st_per_s > 1.5:
                contour = "rising"                 # question-like / open
            elif slope_st_per_s < -1.5:
                contour = "falling"                # closing / decided
        pitch = {
            "detected": True,
            "median_hz": round(median_hz, 1),
            "range_hz": [round(float(np.percentile(voiced, 5)), 1),
                         round(float(np.percentile(voiced, 95)), 1)],
            "variability_semitones_std": round(float(np.std(semis)), 2),
            "final_contour": contour,
            "final_slope_st_per_s": round(slope_st_per_s, 2),
        }

    # --- Loudness dynamics (RMS in dBFS-ish) ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    speech_db = rms_db[rms_db > (rms_db.max() - 40)]      # ignore deep silence
    loudness = {
        "mean_db": round(float(np.mean(speech_db)), 1),
        "dynamic_range_db": round(float(np.percentile(speech_db, 95)
                                        - np.percentile(speech_db, 5)), 1),
    }

    # --- Pauses & speech/silence segmentation ---
    intervals = librosa.effects.split(y, top_db=30)      # non-silent spans
    speech_dur = float(sum((b - a) for a, b in intervals)) / sr
    pauses = []
    for (a0, b0), (a1, _) in zip(intervals[:-1], intervals[1:]):
        gap = (a1 - b0) / sr
        if gap > 0.18:                                   # >180ms counts as a pause
            pauses.append(round(float(gap), 2))
    timing = {
        "total_s": round(dur, 2),
        "speech_s": round(speech_dur, 2),
        "speech_ratio": round(speech_dur / dur, 2) if dur else 0.0,
        "pause_count": len(pauses),
        "longest_pause_s": max(pauses) if pauses else 0.0,
    }

    # --- Speaking-rate proxy (syllable nuclei ≈ energy-envelope peaks) ---
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=5, delta=0.5, wait=4)
    syl_per_s = (len(peaks) / speech_dur) if speech_dur > 0.3 else 0.0
    rate = {
        "syllable_nuclei": int(len(peaks)),
        "rate_syll_per_s_proxy": round(float(syl_per_s), 2),
        "note": "energy-peak proxy, not a true syllable count",
    }

    return {"pitch": pitch, "loudness": loudness,
            "timing": timing, "rate": rate}


# ----------------------------------------------------------------------------- Tier 2
def tier2_emotion(y_16k):
    import torch, torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model, Wav2Vec2PreTrainedModel)

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = self.out_proj(torch.tanh(self.dense(self.dropout(x))))
            return x

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()   # 5.x equivalent of init_weights(); sets all_tied_weights_keys

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            hidden = torch.mean(hidden, dim=1)
            return hidden, self.classifier(hidden)

    global _EMO_CACHE
    if _EMO_CACHE is None:                                   # load once, reuse
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Wav2Vec2Processor.from_pretrained(MODEL)
        model = EmotionModel.from_pretrained(MODEL).to(device).eval()
        _EMO_CACHE = (processor, model, device)
    processor, model, device = _EMO_CACHE

    x = processor(y_16k, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        _, logits = model(x.to(device))
    a, d, v = (float(t) for t in logits[0].cpu().numpy())   # arousal, dominance, valence
    return {
        "model": MODEL,
        "arousal": round(a, 3), "dominance": round(d, 3), "valence": round(v, 3),
        "scale": "≈0..1 (low..high)",
        "device": device,
    }


# ----------------------------------------------------------------------------- interpret
def lvl(x):
    return "low" if x < 0.40 else "high" if x > 0.60 else "moderate"


def interpret(t1, t2):
    lines = []
    p = t1["pitch"]
    if p.get("detected"):
        lines.append(
            f"Pitch sits around {p['median_hz']} Hz with "
            f"{p['variability_semitones_std']} semitone spread "
            f"({'expressive' if p['variability_semitones_std'] > 3 else 'fairly level'}); "
            f"the phrase ends on a {p['final_contour']} contour.")
    r = t1["rate"]["rate_syll_per_s_proxy"]
    tempo = "brisk" if r > 4.5 else "deliberate" if r < 2.8 else "steady"
    tm = t1["timing"]
    lines.append(
        f"Delivery is {tempo} (~{r}/s), {tm['speech_ratio']*100:.0f}% voiced, "
        f"{tm['pause_count']} notable pause(s)"
        + (f", longest {tm['longest_pause_s']}s." if tm['pause_count'] else "."))
    if t2:
        lines.append(
            f"Emotion estimate — arousal {lvl(t2['arousal'])} ({t2['arousal']}), "
            f"valence {lvl(t2['valence'])} ({t2['valence']}), "
            f"dominance {lvl(t2['dominance'])} ({t2['dominance']}). "
            f"[population-trained estimate — treat as a hint, not a verdict]")
    return lines


# ----------------------------------------------------------------------------- main
def _features(res):
    """Flatten a result dict into the comparable feature vector."""
    t1 = res["tier1_prosody"]; t2 = res.get("tier2_emotion") or {}
    tm = t1["timing"]
    return {
        "rate": t1["rate"]["rate_syll_per_s_proxy"],
        "speech_ratio": tm["speech_ratio"],
        "pauses_per_min": round(tm["pause_count"] / max(tm["total_s"], 1e-6) * 60, 1),
        "arousal": t2.get("arousal"), "valence": t2.get("valence"),
        "dominance": t2.get("dominance"),
        "pitch_hz": t1["pitch"].get("median_hz"),
    }


def match_baseline(res, baseline):
    """Classify a clip against the baseline modes and describe it vs personal-normal."""
    modes = baseline.get("modes", {})
    if not modes:
        return None
    clip = _features(res)
    mfeat = {m: _features(r) for m, r in modes.items()}
    spread = {}
    for k in MATCH_KEYS:                       # normalise by spread across modes
        vals = [mf[k] for mf in mfeat.values() if mf[k] is not None]
        spread[k] = (max(vals) - min(vals)) if len(vals) > 1 and max(vals) != min(vals) else 1.0
    dist = {}
    for m, mf in mfeat.items():
        acc = n = 0
        for k in MATCH_KEYS:
            if clip[k] is None or mf[k] is None:
                continue
            acc += ((clip[k] - mf[k]) / spread[k]) ** 2; n += 1
        dist[m] = round((acc / n) ** 0.5, 2) if n else 9.99
    nearest = min(dist, key=dist.get)

    ref = baseline.get("personal_normal", {}).get("reference_mode", "Neutral")
    neu = mfeat.get(ref)
    notes = []
    if neu:
        dr = clip["rate"] - neu["rate"]
        if abs(dr) > 0.5:
            notes.append(f"{'faster' if dr > 0 else 'slower'} ({clip['rate']} vs {neu['rate']}/s)")
        dp = clip["pauses_per_min"] - neu["pauses_per_min"]
        if abs(dp) > 8:
            notes.append("more hesitant" if dp > 0 else "more fluent")
        if clip["valence"] is not None and neu["valence"] is not None:
            dv = clip["valence"] - neu["valence"]
            if abs(dv) > 0.12:
                notes.append("brighter tone" if dv > 0 else "flatter tone")
    verdict = f"closest to your '{nearest}' voice"
    if notes:
        verdict += " — " + ", ".join(notes) + f" vs {ref.lower()}"
    return {"nearest_mode": nearest, "distances": dist, "verdict": verdict}


def analyze(y, sr, do_tier2=True):
    t1 = tier1_prosody(y, sr)
    t2 = None
    if do_tier2:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        try:
            t2 = tier2_emotion(y16)
        except Exception as e:
            print(f"[!] Tier 2 failed ({e.__class__.__name__}: {e})")
    return {"tier1_prosody": t1, "tier2_emotion": t2,
            "interpretation": interpret(t1, t2)}


def auto_boundaries(y, sr, n_segments):
    """Find the (n_segments-1) largest silence gaps as split points (seconds)."""
    iv = librosa.effects.split(y, top_db=30)
    gaps = []
    for (a0, b0), (a1, _) in zip(iv[:-1], iv[1:]):
        gaps.append(((b0 + a1) / 2 / sr, (a1 - b0) / sr))   # (midpoint_s, gap_s)
    gaps.sort(key=lambda g: -g[1])
    cuts = sorted(m for m, _ in gaps[:max(0, n_segments - 1)])
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--no-tier2", action="store_true", help="skip the emotion model")
    ap.add_argument("--hpf", type=float, default=HPF_DEFAULT,
                    help="high-pass cutoff Hz (0 disables)")
    ap.add_argument("--notch", type=float, default=50.0,
                    help="mains-hum fundamental to notch + harmonics (0 disables)")
    ap.add_argument("--baseline", nargs="*", metavar="MODE",
                    help="treat clip as a calibration baseline; split into the named "
                         "modes (default: Neutral Animated Tired)")
    ap.add_argument("--boundaries",
                    help="split points 't1,t2' OR explicit spans 'a-b,c-d,e-f' "
                         "(spans let you drop transition gaps)")
    ap.add_argument("--baseline-file", default=DEFAULT_BASELINE,
                    help="baseline JSON to classify a single clip against")
    ap.add_argument("--no-baseline-match", action="store_true")
    ap.add_argument("--windows", type=float, metavar="SECONDS",
                    help="long-recording mode: emotional/prosodic timeline in "
                         "fixed windows of this length")
    args = ap.parse_args()

    path = os.path.abspath(args.audio)
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")

    y, sr = librosa.load(path, sr=None, mono=True)
    y = hum_notch(y, sr, args.notch)
    y = high_pass(y, sr, args.hpf)
    print(f"[*] {os.path.basename(path)} — {len(y)/sr:.2f}s @ {sr} Hz "
          f"(notch {args.notch:g}Hz+harmonics, high-pass {args.hpf:g}Hz)")
    do_t2 = not args.no_tier2
    if do_t2:
        print("[*] Tier 2 emotion model on (first run downloads ~1.2GB)…")

    # -------------------------------------------------- baseline (multi-mode) mode
    if args.baseline is not None:
        modes = args.baseline or ["Neutral", "Animated", "Tired"]
        if args.boundaries and "-" in args.boundaries:
            spans = [tuple(float(t) for t in s.split("-"))
                     for s in args.boundaries.split(",")]            # explicit a-b,c-d
            cuts = []
        elif args.boundaries:
            cuts = [float(x) for x in args.boundaries.split(",")]
            edges = [0.0] + cuts + [len(y) / sr]
            spans = list(zip(edges[:-1], edges[1:]))
        else:
            cuts = auto_boundaries(y, sr, len(modes))
            edges = [0.0] + cuts + [len(y) / sr]
            spans = list(zip(edges[:-1], edges[1:]))
        segs = {}
        for name, (t0, t1) in zip(modes, spans):
            seg = y[int(t0 * sr):int(t1 * sr)]
            print(f"\n--- {name}: {t0:.1f}–{t1:.1f}s ({t1-t0:.1f}s) ---")
            res = analyze(seg, sr, do_t2)
            res["span_s"] = [round(t0, 2), round(t1, 2)]
            for line in res["interpretation"]:
                print("   •", line)
            segs[name] = res
        # personal-normal reference (built from the Neutral mode if present)
        ref_mode = "Neutral" if "Neutral" in segs else modes[0]
        rp = segs[ref_mode]["tier1_prosody"]["pitch"]
        normal = {"reference_mode": ref_mode,
                  "median_hz": rp.get("median_hz"),
                  "pitch_range_hz": rp.get("range_hz"),
                  "rate_syll_per_s": segs[ref_mode]["tier1_prosody"]["rate"]["rate_syll_per_s_proxy"]}
        out = os.path.splitext(path)[0] + ".baseline.json"
        with open(out, "w") as f:
            json.dump({"schema": "aethel.baseline/0.1", "source": path,
                       "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                       "high_pass_hz": args.hpf, "notch_hz": args.notch,
                       "spans_s": [[round(a, 2), round(b, 2)] for a, b in spans],
                       "personal_normal": normal, "modes": segs}, f, indent=2)
        print(f"\n[*] baseline written: {out}")
        return

    # -------------------------------------------------- windowed timeline mode
    if args.windows:
        win = int(args.windows * sr)
        tl = []
        n = max(1, int(np.ceil(len(y) / win)))
        for i in range(n):
            seg = y[i * win:(i + 1) * win]
            if len(seg) < 0.5 * sr:           # ignore a tiny trailing sliver
                continue
            t0 = i * args.windows
            r = analyze(seg, sr, do_t2)
            f = _features(r)
            f["t"] = round(t0, 1)
            tl.append(f)
            print(f"  {t0:6.1f}s  rate={f['rate']:4.1f}  voiced={f['speech_ratio']*100:3.0f}%"
                  + (f"  A={f['arousal']:.2f} V={f['valence']:.2f} D={f['dominance']:.2f}"
                     if f['arousal'] is not None else ""))
        out = os.path.splitext(path)[0] + ".timeline.json"
        with open(out, "w") as fh:
            json.dump({"schema": "aethel.timeline/0.1", "source": path,
                       "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                       "window_s": args.windows, "high_pass_hz": args.hpf,
                       "notch_hz": args.notch, "windows": tl}, fh, indent=2)
        # arc summary
        def arc(key):
            vals = [(w["t"], w[key]) for w in tl if w.get(key) is not None]
            if not vals:
                return None
            lo = min(vals, key=lambda x: x[1]); hi = max(vals, key=lambda x: x[1])
            mean = sum(v for _, v in vals) / len(vals)
            return {"mean": round(mean, 3), "min": [lo[0], round(lo[1], 3)],
                    "max": [hi[0], round(hi[1], 3)]}
        print("\n=== ARC ===")
        for k in ("arousal", "valence", "dominance", "rate"):
            a = arc(k)
            if a:
                print(f"  {k:9}: mean {a['mean']}  peak {a['max'][1]}@{a['max'][0]}s  "
                      f"low {a['min'][1]}@{a['min'][0]}s")
        print(f"\n[*] timeline written: {out}")
        return

    # -------------------------------------------------- single-clip mode
    res = analyze(y, sr, do_t2)
    bmatch = None
    if not args.no_baseline_match and os.path.isfile(args.baseline_file):
        try:
            with open(args.baseline_file) as bf:
                bmatch = match_baseline(res, json.load(bf))
        except Exception as e:
            print(f"[!] baseline match skipped ({e.__class__.__name__}: {e})")
    out = os.path.splitext(path)[0] + ".prosody.json"
    with open(out, "w") as f:
        json.dump({"schema": "aethel.prosody/0.1", "source": path,
                   "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "high_pass_hz": args.hpf, "notch_hz": args.notch,
                   "baseline_match": bmatch, **res}, f, indent=2)
    print("\n=== INTERPRETATION ===")
    for line in res["interpretation"]:
        print("  •", line)
    if bmatch:
        print("  ◆", bmatch["verdict"])
    print(f"\n[*] sidecar written: {out}")


if __name__ == "__main__":
    main()
