"""Voice analysis: measures every character's voice from the stretches where they talk alone and
writes the report (Executive Summary, Sections 1-4 per vocalist, Separation Matrix).

All numbers are measured from the audio. Labels are derived from the numbers with fixed rules, so
the same voice always gets the same description. Ranges are for the SPEAKING voice."""
import math

import numpy as np
from scipy.signal import resample_poly

from .pitch import yin

SR16 = 16000
C_SOUND = 35000.0                         # cm/s, warm humid air in the vocal tract
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note(f):
    if not f or f <= 0:
        return "-"
    m = int(round(69 + 12 * math.log2(f / 440.0)))
    return "%s%d" % (NOTES[m % 12], m // 12 - 1)


def st(a, b):
    return 12 * math.log2(a / b) if a > 0 and b > 0 else 0.0


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def lpc(frame, order):
    """Autocorrelation LPC (Levinson-Durbin)."""
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 0:
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= (1 - k * k)
        if e <= 0:
            return None
    return a


def formants(x16, f0_frames, hop=160):
    """Median F1-F3 (Hz) over voiced frames, LPC at 8 kHz."""
    x8 = resample_poly(x16, 1, 2)
    x8 = np.append(x8[0], x8[1:] - 0.63 * x8[:-1])                 # pre-emphasis
    win = 200                                                       # 25 ms at 8 kHz
    h = np.hamming(win)
    F = []
    for j in np.nonzero(f0_frames > 0)[0][::2]:
        c = j * hop // 2
        fr = x8[c - win // 2:c + win // 2]
        if len(fr) < win:
            continue
        a = lpc(fr * h, 10)
        if a is None:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0.01]
        fr_hz = np.angle(roots) * 8000 / (2 * np.pi)
        bw = -np.log(np.abs(roots)) * 8000 / np.pi
        ok = (fr_hz > 200) & (fr_hz < 3800) & (bw < 450)
        cand = np.sort(fr_hz[ok])
        if len(cand) >= 3 and cand[0] < 1100:
            F.append(cand[:3])
    if len(F) < 5:
        return None
    return np.median(np.array(F), 0)


def harm_amp(spec, freqs, f, tol):
    m = (freqs > f - tol) & (freqs < f + tol)
    return 20 * np.log10(spec[m].max() + 1e-12) if m.any() else -240.0


class Profiler:
    def __init__(self, project):
        self.p = project

    # ------------------------------------------------------------------ material
    def _solo_frames(self, k):
        p = self.p
        hard = p.hard[k] > 0.9
        return hard & (p.o < 0.05) & (p.sw > 0.9)

    def _active_runs(self, k):
        e = self.p.lane_energy[k]
        live = 10 * np.log10(e + 1e-30) > 10 * np.log10(self.p.lane_energy.max() + 1e-30) - 45
        from .engine import drop_short, fill_gaps
        live = drop_short(fill_gaps(live, 30), 20)
        return runs(live)

    # ------------------------------------------------------------------ measurements
    def measure(self, k, max_s=150):
        m = self._measure(k, max_s, False)
        if m is None or m["solo_seconds"] < 2.0:
            m2 = self._measure(k, max_s, True)
            if m2 is not None and (m is None or m2["voiced_frames"] > m["voiced_frames"]):
                m2["from_stem"] = True
                return m2
        return m

    def _measure(self, k, max_s, from_stem):
        p = self.p
        if from_stem:
            import soundfile as sf
            st48 = sf.read(p.stem_paths()[k], dtype="float32", always_2d=True)[0].mean(1)
            x16_all = resample_poly(st48, 1, 3).astype(np.float32)
            solo = np.zeros(p.nF, bool)
            for a, b in self._active_runs(k):
                solo[a:b] = True
        else:
            st48 = p.x.mean(1)
            x16_all = p.mono16
            solo = self._solo_frames(k)
        rr = [(a, b) for a, b in runs(solo) if b - a >= 20]
        tot = 0
        sel = []
        for a, b in sorted(rr, key=lambda r: -(r[1] - r[0])):
            if tot >= max_s * 100:
                break
            sel.append((a, b))
            tot += b - a
        sel.sort()
        f0s, pers, rms16 = [], [], []
        f0_runs = []
        formant_x, formant_f = [], []
        H12, tilt, ring, a1p0, hf_ratio = [], [], [], [], []
        win_yin = int(SR16 / 55.0 * 2.2)
        shift = int(round((win_yin / 2 - 80) / 160))
        x48 = st48
        hop48 = p.hop
        nfft = 4096
        freqs = np.fft.rfftfreq(nfft, 1 / p.sr)
        for a, b in sel:
            x = x16_all[a * 160:b * 160]
            f, pr = yin(x, SR16, 160)
            f0_frames = np.zeros(b - a)
            per_frames = np.zeros(b - a)
            j = np.arange(len(f)) + shift
            m = j < b - a
            f0_frames[j[m]] = f[m]
            per_frames[j[m]] = pr[m]
            f0s.append(f0_frames)
            pers.append(per_frames)
            f0_runs.append(f0_frames)
            frm = (x[:len(x) // 160 * 160].reshape(-1, 160).astype(np.float64) ** 2).mean(1)
            rms16.append(frm)
            fm = formants(x, f0_frames)
            if fm is not None:
                formant_f.append(fm)
                formant_x.append(b - a)
            # spectra of voiced and unvoiced frames, full bandwidth from the original
            for jj in np.nonzero(f0_frames > 0)[0][::4]:
                c = (a + jj) * hop48 + hop48 // 2
                seg = x48[c - nfft // 2:c + nfft // 2]
                if len(seg) < nfft:
                    continue
                sp = np.abs(np.fft.rfft(seg * np.hanning(nfft)))
                f0 = f0_frames[jj]
                H12.append(harm_amp(sp, freqs, f0, f0 * 0.2) - harm_amp(sp, freqs, 2 * f0, f0 * 0.2))
                band = (freqs > 500) & (freqs < 4000)
                lf = np.log2(freqs[band])
                ld = 20 * np.log10(sp[band] + 1e-12)
                tilt.append(np.polyfit(lf, ld, 1)[0])
                e_lo = (sp[(freqs > 80) & (freqs < 1000)] ** 2).sum()
                e_ring = (sp[(freqs > 2000) & (freqs < 4000)] ** 2).sum()
                ring.append(10 * np.log10(e_ring / (e_lo + 1e-20) + 1e-20))
                bnd = lambda lo, hi: 10 * np.log10((sp[(freqs > lo) & (freqs < hi)] ** 2).mean() + 1e-24)
                a1p0.append((bnd(200, 320) + bnd(1800, 2400)) / 2 - bnd(700, 1200))
            for jj in np.nonzero((f0_frames == 0) & (frm[:len(f0_frames)] > frm.max() * 1e-3 if len(frm) else False))[0][::3]:
                c = (a + jj) * hop48 + hop48 // 2
                seg = x48[c - 1024:c + 1024]
                if len(seg) < 2048:
                    continue
                sp = np.abs(np.fft.rfft(seg * np.hanning(2048))) ** 2
                fq = np.fft.rfftfreq(2048, 1 / p.sr)
                hf_ratio.append(10 * np.log10(sp[(fq > 4000) & (fq < 10000)].sum() / (sp[fq < 4000].sum() + 1e-20) + 1e-20))
        if not f0s:
            return None
        F0 = np.concatenate(f0s)
        PER = np.concatenate(pers)
        V = F0 > 0
        if V.sum() < 30:
            return None
        fv = F0[V]
        med = float(np.median(fv))
        p5, p95 = float(np.percentile(fv, 5)), float(np.percentile(fv, 95))
        # perturbation (frame level): consecutive voiced frames
        jit, shim = [], []
        for f0r, rr16 in zip(f0_runs, rms16):
            for a2, b2 in runs(f0r > 0):
                if b2 - a2 < 5:
                    continue
                t = 1 / f0r[a2:b2]
                jit.append(np.mean(np.abs(np.diff(t))) / np.mean(t))
                amp = np.sqrt(rr16[a2:min(b2, len(rr16))])
                if len(amp) > 4:
                    shim.append(np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + 1e-12))
        hnr = 10 * np.log10(np.median(np.clip(PER[V], 0.01, 0.999)) / (1 - np.median(np.clip(PER[V], 0.01, 0.999))))
        fry = float(((F0 > 0) & (F0 < 72)).sum() / max(1, V.sum()))
        # vibrato: 4-8 Hz modulation of F0 within voiced stretches of 0.5 s or more
        vib = []
        for f0r in f0_runs:
            for a2, b2 in runs(f0r > 0):
                if b2 - a2 < 50:
                    continue
                c = 1200 * np.log2(f0r[a2:b2] / np.median(f0r[a2:b2]))
                c = c - np.polyval(np.polyfit(np.arange(len(c)), c, 2), np.arange(len(c)))
                S = np.abs(np.fft.rfft(c * np.hanning(len(c)), 512))
                fq = np.fft.rfftfreq(512, 0.01)
                band = (fq >= 4) & (fq <= 8)
                ext = float(np.percentile(c, 95) - np.percentile(c, 5))
                if S[band].max() > 5 * np.median(S[(fq > 1) & (fq < 15)]) and 20 <= ext <= 220:
                    rate = float(fq[band][S[band].argmax()])
                    vib.append((rate, ext, a2))
        # glides and breaks
        gliss = breaks = 0
        for f0r in f0_runs:
            for a2, b2 in runs(f0r > 0):
                seg = 12 * np.log2(f0r[a2:b2] / 100.0)
                if len(seg) > 25:
                    d25 = np.abs(seg[25:] - seg[:-25])
                    gliss += int((d25 > 5).any())
                if len(seg) > 1:
                    breaks += int((np.abs(np.diff(seg)) > 7).sum())
        # dynamics on 400 ms windows across all of this character's talking
        lev = []
        stem = None
        try:
            import soundfile as sf
            stem = sf.read(p.stem_paths()[k], dtype="float32", always_2d=True)[0].mean(1)
        except Exception:  # noqa: BLE001
            stem = None
        if stem is not None:
            w = int(0.4 * p.sr)
            for a, b in self._active_runs(k):
                y = stem[a * hop48:b * hop48]
                for q in range(0, len(y) - w, w):
                    lev.append(10 * np.log10((y[q:q + w] ** 2).mean() + 1e-12))
        lev = np.array(lev) if lev else np.array([-60.0])
        lev = lev[lev > lev.max() - 40]
        # speech rate: syllable nuclei per second of solo talk
        syl = 0
        secs = 0
        from scipy.signal import find_peaks
        for rr16 in rms16:
            env = 10 * np.log10(rr16 + 1e-12)
            env = np.convolve(env, np.ones(5) / 5, "same")
            pk, _ = find_peaks(env, prominence=4, distance=10, height=env.max() - 30)
            syl += len(pk)
            secs += len(rr16) / 100
        fmt = None
        if formant_f:
            fmt = np.average(np.array(formant_f), axis=0, weights=np.array(formant_x, float))
        # uniform-tube fit to F2 and F3 (F1 depends too much on the vowel)
        vtl = float(C_SOUND / 4 * 34 / (3 * fmt[1] + 5 * fmt[2])) if fmt is not None else None
        return {
            "f0_median": med, "f0_p5": p5, "f0_p95": p95, "f0_range_st": st(p95, p5),
            "formants": [float(v) for v in fmt] if fmt is not None else None, "vtl_cm": vtl,
            "hnr_db": float(hnr), "jitter_pct": float(100 * np.median(jit)) if jit else None,
            "shimmer_pct": float(100 * np.median(shim)) if shim else None,
            "h1h2_db": float(np.median(H12)) if H12 else None,
            "tilt_db_oct": float(np.median(tilt)) if tilt else None,
            "ring_db": float(np.median(ring)) if ring else None,
            "a1p0_db": float(np.median(a1p0)) if a1p0 else None,
            "hf_consonant_db": float(np.median(hf_ratio)) if hf_ratio else None,
            "fry_pct": 100 * fry, "vibrato": vib, "glides": gliss, "breaks": breaks,
            "level_p10": float(np.percentile(lev, 10)), "level_p50": float(np.percentile(lev, 50)),
            "level_p95": float(np.percentile(lev, 95)),
            "syllables_per_s": syl / secs if secs else None, "solo_seconds": tot / 100,
            "voiced_frames": int(V.sum()), "from_stem": False,
        }

    # ------------------------------------------------------------------ labels
    @staticmethod
    def classify(m):
        L = {}
        f0, vtl = m["f0_median"], m["vtl_cm"]
        if f0 > 260 and (vtl is None or vtl < 13.5):
            sex = "High / child-like voice"
        elif f0 < 150:
            sex = "Male" if (vtl is None or vtl >= 15.0) else "Androgynous / ambiguous"
        elif f0 > 165:
            sex = "Female" if (vtl is None or vtl < 16.8) else "Androgynous / ambiguous"
        elif vtl is not None:
            sex = "Male" if vtl >= 16.3 else "Female" if vtl < 15.3 else "Androgynous / ambiguous"
        else:
            sex = "Androgynous / ambiguous"
        L["sex"] = sex
        if sex == "Male":
            sub = "Bass" if f0 < 100 else ("Baritone" if f0 < 135 else "Tenor")
        elif sex == "Female":
            sub = "Contralto" if f0 < 185 else ("Mezzo-soprano" if f0 < 225 else "Soprano")
        else:
            sub = "Overlapping range"
        L["sub"] = sub
        ind = []
        if vtl is not None:
            ind.append("estimated vocal tract length ~%.1f cm (%s)" % (vtl, "long" if vtl >= 16.5 else "short" if vtl < 14.8 else "medium"))
        if m["formants"]:
            F1, F2, F3 = m["formants"]
            ind.append("mean formant spacing ~%.0f Hz" % ((F3 - F1) / 2))
        if m["tilt_db_oct"] is not None:
            ind.append("chest resonance prominent" if m["tilt_db_oct"] < -6 else "head/nasal resonance prominent"
                       if m["tilt_db_oct"] > -2.5 else "balanced chest/head resonance")
        if sex == "Androgynous / ambiguous":
            ind.append("pitch and resonance point to different ranges (mixed resonance profile)")
        L["s1_ind"] = ind
        # Section 2
        h12, tilt = m["h1h2_db"] or 0, m["tilt_db_oct"] or -7
        vt = vtl or 15.5
        L["weight"] = ("Heavy / dense" if (h12 < 2 and vt >= 16.3) else
                       "Light / thin" if (h12 > 6 or vt < 14.5) else "Medium")
        hnr, jit, shim = m["hnr_db"], m["jitter_pct"] or 0, m["shimmer_pct"] or 0
        tex = []
        if hnr >= 12 and jit < 1.5:
            tex.append("Clean / smooth")
        if h12 > 6 and hnr < 12:
            tex.append("Breathy")
        if jit >= 2.0 or (hnr < 8 and shim > 15):
            tex.append("Raspy / gravelly")
        if hnr < 7 and m["level_p95"] > -20:
            tex.append("Grit (loud, rough phonation)")
        if not tex:
            tex.append("Moderately clean")
        L["texture"] = tex
        L["nasality"] = "-"            # set relative to the other voices in analyse()
        L["fry"] = ("Frequent vocal fry" if m["fry_pct"] > 8 else "Occasional vocal fry" if m["fry_pct"] > 2 else "Little or no vocal fry")
        # Section 3
        L["intonation"] = ("Wide, expressive inflection" if m["f0_range_st"] > 12 else
                           "Moderate inflection" if m["f0_range_st"] > 6 else "Narrow / monotone delivery")
        if m["vibrato"]:
            rate = np.median([v[0] for v in m["vibrato"]])
            ext = np.median([v[1] for v in m["vibrato"]])
            L["vibrato"] = "%s (%.1f Hz), %s (%.0f cents), present in %d sustained note(s)" % (
                "Fast" if rate > 6 else "Slow", rate, "wide" if ext > 100 else "narrow/tight", ext, len(m["vibrato"]))
        else:
            L["vibrato"] = "None - straight tone (normal for speech)"
        lv = m["level_p50"]
        dyn = ("Fortissimo" if lv > -16 else "Forte" if lv > -22 else "Mezzo-forte" if lv > -28 else
               "Mezzo-piano" if lv > -34 else "Piano" if lv > -40 else "Pianissimo")
        L["dynamics"] = "%s on average; range %.0f dB (quietest %.0f dBFS, loudest %.0f dBFS)" % (
            dyn, m["level_p95"] - m["level_p10"], m["level_p10"], m["level_p95"])
        # Section 4
        hf = m["hf_consonant_db"]
        L["enunciation"] = "-" if hf is None else ("Crisp / clinical consonants" if hf > -6 else
                                                   "Clear consonants" if hf > -14 else "Soft / elided consonants")
        if f0 > (300 if sex == "Female" else 200) and h12 > 6:
            plc = "Head voice / falsetto"
        elif (m["ring_db"] or -30) > -12:
            plc = "Mask / twang (bright 2-4 kHz focus)"
        else:
            plc = "Chest voice (grounded, speech-like)"
        L["placement"] = plc
        sty = ["Spoken-word delivery"]
        if m["glides"]:
            sty.append("glissando-like pitch slides (%d)" % m["glides"])
        if m["breaks"]:
            sty.append("vocal breaks / pitch jumps (%d)" % m["breaks"])
        if m["vibrato"] and jit > 1.5:
            sty.append("possible weeping / tremulous tone")
        L["style"] = sty
        return L


def _tc(sec, start, fps):
    sec = max(0, sec + start)
    fr = int(sec * fps + 1e-6)
    n = int(round(fps))
    f = fr % n
    s = fr // n
    return "%02d:%02d:%02d:%02d" % ((s // 3600) % 24, (s // 60) % 60, s % 60, f)


def analyse(project, fps=25.0):
    """Returns (data dict, markdown report)."""
    p = project
    pr = Profiler(p)
    start = (p.time_ref or 0) / p.sr
    people = []
    for k, ch in enumerate(p.chars):
        m = pr.measure(k)
        rr = pr._active_runs(k)
        talk = sum(b - a for a, b in rr) / 100
        ovl = float(((p.lane_energy[k] > 0) & (p.o > 0.5)).sum() / 100) if hasattr(p, "o") else 0
        people.append({"id": ch["id"], "name": ch["name"], "m": m, "L": Profiler.classify(m) if m else None,
                       "runs": [(a / 100, b / 100) for a, b in rr], "talk_s": talk,
                       "voice": ch["c"]})
    idx = [(q["m"]["a1p0_db"], q) for q in people if q["m"] and q["m"]["a1p0_db"] is not None]
    if len(idx) >= 2:
        vals = np.array([v for v, _ in idx])
        for v, q in idx:
            z = (v - vals.mean()) / (vals.std() + 1e-6)
            q["L"]["nasality"] = ("More nasal than the others in this scene" if z > 0.8 else
                                  "Less nasal than the others in this scene" if z < -0.8 else "Average nasality for this scene")
    md = []
    md.append("# Un-tie Voice Analysis")
    md.append("Source: %s  \nDuration: %.1f s, %d Hz, %s" % (
        p.path, p.N / p.sr, p.sr, "mono" if p.C == 1 else "stereo" if p.C == 2 else "%d ch" % p.C))
    md.append("All ranges refer to the speaking voice (measured from stretches where each character talks alone).")
    md.append("\n## 1. Executive Summary")
    md.append("**Distinct vocalists detected: %d**\n" % len(people))
    md.append("| Vocalist | Classification | Talk time | Appears (timecode) |")
    md.append("|---|---|---|---|")
    for q in people:
        cls = ("%s - %s" % (q["L"]["sex"], q["L"]["sub"])) if q["L"] else "not enough solo material"
        shown = q["runs"][:12]
        ts = ", ".join("%s-%s" % (_tc(a, start, fps), _tc(b, start, fps)) for a, b in shown)
        if len(q["runs"]) > 12:
            ts += " (+%d more)" % (len(q["runs"]) - 12)
        md.append("| %s | %s | %.1f s | %s |" % (q["name"], cls, q["talk_s"], ts))
    if hasattr(p, "o"):
        ovs = runs(p.o > 0.5)
        md.append("\nOverlapping speech: %d stretch(es), %.1f s in total." % (len(ovs), sum(b - a for a, b in ovs) / 100))

    md.append("\n## 2. Detailed Breakdown per Vocalist")
    for q in people:
        m, L = q["m"], q["L"]
        md.append("\n### %s" % q["name"])
        if not m:
            md.append("Not enough solo dialogue to measure this voice reliably (needs a few seconds where this character talks alone).")
            continue
        fm = m["formants"]
        md.append("Measured from %.1f s of %s." % (m["solo_seconds"], "this character's separated stem (little solo dialogue, less reliable)"
                                                   if m.get("from_stem") else "solo dialogue"))
        md.append("\n**Section 1 - Primary Biological Classification**")
        md.append("- Classification: **%s**, sub-classification **%s**" % (L["sex"], L["sub"]))
        md.append("- Fundamental frequency (F0): median %.0f Hz (%s), range %.0f-%.0f Hz (%s-%s, %.1f semitones)" % (
            m["f0_median"], note(m["f0_median"]), m["f0_p5"], m["f0_p95"], note(m["f0_p5"]), note(m["f0_p95"]), m["f0_range_st"]))
        if fm:
            md.append("- Formants: F1 %.0f Hz, F2 %.0f Hz, F3 %.0f Hz" % tuple(fm))
        md.append("- Indicators: " + "; ".join(L["s1_ind"]))
        md.append("\n**Section 2 - Vocal Anatomy & Texture**")
        md.append("- Vocal weight: %s (H1-H2 %s dB, spectral tilt %s dB/octave)" % (
            L["weight"], "%.1f" % m["h1h2_db"] if m["h1h2_db"] is not None else "-",
            "%.1f" % m["tilt_db_oct"] if m["tilt_db_oct"] is not None else "-"))
        md.append("- Texture: %s (harmonics-to-noise %.1f dB, jitter %s%%, shimmer %s%%)" % (
            ", ".join(L["texture"]), m["hnr_db"], "%.2f" % m["jitter_pct"] if m["jitter_pct"] is not None else "-",
            "%.1f" % m["shimmer_pct"] if m["shimmer_pct"] is not None else "-"))
        md.append("- Nasality: %s (nasal index %s dB: 1 kHz dip relative to 250 Hz / 2 kHz)" % (
            L["nasality"], "%.1f" % m["a1p0_db"] if m["a1p0_db"] is not None else "-"))
        md.append("- Vocal fry: %s (%.1f%% of voiced frames below 72 Hz)" % (L["fry"], m["fry_pct"]))
        md.append("\n**Section 3 - Pitch, Vibrato & Intonation**")
        md.append("- Pitch / intonation: %s; speech is inflected rather than pitch-centred" % L["intonation"])
        md.append("- Vibrato: %s" % L["vibrato"])
        md.append("- Dynamic range: %s" % L["dynamics"])
        md.append("\n**Section 4 - Articulation, Diction & Stylistic Delivery**")
        md.append("- Enunciation: %s (consonant energy 4-10 kHz %s dB relative to below 4 kHz)" % (
            L["enunciation"], "%.1f" % m["hf_consonant_db"] if m["hf_consonant_db"] is not None else "-"))
        md.append("- Speech rate: %s syllables per second" % ("%.1f" % m["syllables_per_s"] if m["syllables_per_s"] else "-"))
        md.append("- Placement / resonance focus: %s (2-4 kHz ring %s dB)" % (
            L["placement"], "%.1f" % m["ring_db"] if m["ring_db"] is not None else "-"))
        md.append("- Stylistic delivery: %s" % ", ".join(L["style"]))

    md.append("\n## 3. Multi-Vocalist Separation Matrix")
    md.append("| Vocalist | Sex | Sub-class | F0 median | Vocal tract | Weight | Texture | Placement | Nasality |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for q in people:
        m, L = q["m"], q["L"]
        if not m:
            md.append("| %s | - | - | - | - | - | - | - | - |" % q["name"])
            continue
        md.append("| %s | %s | %s | %.0f Hz (%s) | %s | %s | %s | %s | %s |" % (
            q["name"], L["sex"], L["sub"], m["f0_median"], note(m["f0_median"]),
            "%.1f cm" % m["vtl_cm"] if m["vtl_cm"] else "-", L["weight"], ", ".join(L["texture"]),
            L["placement"].split(" (")[0], L["nasality"]))
    pairs = []
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if not (a["m"] and b["m"]):
                continue
            gap = abs(st(a["m"]["f0_median"], b["m"]["f0_median"]))
            vt = abs((a["m"]["vtl_cm"] or 0) - (b["m"]["vtl_cm"] or 0)) if a["m"]["vtl_cm"] and b["m"]["vtl_cm"] else None
            cos = float(np.dot(a["voice"], b["voice"]))
            if a["L"]["sex"] != b["L"]["sex"] and gap > 4:
                rate = "High"
            elif gap > 3 or cos < 0.35:
                rate = "Medium"
            else:
                rate = "Low - similar voices; use a higher Splitting Intensity and check overlaps by ear"
            pairs.append("| %s vs %s | %.1f semitones | %s | %.2f | %s |" % (
                a["name"], b["name"], gap, "%.1f cm" % vt if vt is not None else "-", cos, rate))
    if pairs:
        md.append("\n**Pairwise separability** (how easy each pair is to keep apart)\n")
        md.append("| Pair | Pitch gap | Vocal tract difference | Voice-print similarity (0-1) | Separability |")
        md.append("|---|---|---|---|---|")
        md += pairs
    data = [{"id": q["id"], "name": q["name"], "metrics": q["m"], "labels": q["L"], "runs": q["runs"],
             "talk_s": q["talk_s"]} for q in people]
    return data, "\n".join(md) + "\n"
