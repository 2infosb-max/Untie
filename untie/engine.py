"""Un-tie engine: turns one mixed dialogue track into one stem per character.

How it works
1. Find where people are talking (energy-based voice activity).
2. Run a voice-separation network over every talking stretch (8 s pieces, 2 s overlap).
3. Recognise who each separated voice is, every 0.25 s, with a speaker-identity network.
4. Decide frame by frame (10 ms): silence, one character alone, or an overlap.
5. Build the stems from the ORIGINAL full-resolution recording:
   - one character alone  -> that character gets the original sound untouched (a plain fade, no filtering)
   - overlap              -> the original's spectrum is shared out using the separated voices
   - silence              -> the Residual stem
   The shares always add up to exactly 1, so all stems summed = the original (null test).
"""
import math
import os
import threading
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from . import audio_io
from .models import SR16, WIN16

HOP16, NFFT16, PAD16 = 160, 640, 240
FPS = 100
WSTEP16 = 4000                     # voice identity every 0.25 s
OVERLAP_S = 2.0
BLOCK = 1024                       # frames per render block
COLORS = ["#4FC3F7", "#FF8A65", "#AED581", "#CE93D8", "#FFD54F", "#4DB6AC", "#F48FB1",
          "#9FA8DA", "#FFAB91", "#80CBC4", "#E6EE9C", "#B0BEC5"]


def resample(x, a, b, axis=0):
    if a == b:
        return np.asarray(x, np.float32)
    g = math.gcd(int(a), int(b))
    return resample_poly(x, b // g, a // g, axis=axis).astype(np.float32)


def frame_energy(x, n_frames, hop):
    y = np.zeros(n_frames * hop, np.float32)
    k = min(len(x), len(y))
    y[:k] = x[:k]
    return (y.reshape(n_frames, hop).astype(np.float64) ** 2).mean(1)


def db(e):
    return 10 * np.log10(np.maximum(e, 1e-12))


def runs(mask):
    """[(start, end)] of True stretches."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_gaps(mask, max_gap):
    m = mask.copy()
    r = runs(~m)
    for a, b in r:
        if a > 0 and b < len(m) and b - a <= max_gap:
            m[a:b] = True
    return m


def drop_short(mask, min_len):
    m = mask.copy()
    for a, b in runs(m):
        if b - a < min_len:
            m[a:b] = False
    return m


def dilate(mask, before, after):
    m = np.asarray(mask, bool)
    out = m.copy()
    for a, b in runs(m):
        out[max(0, a - before):min(len(m), b + after)] = True
    return out


def ramp(x, n):
    """Smooth a 0/1 curve into a fade of n frames."""
    return uniform_filter1d(np.asarray(x, np.float32), size=max(1, n), mode="nearest")


def majority(labels, k, size):
    oh = np.zeros((k, len(labels)), np.float32)
    oh[labels, np.arange(len(labels))] = 1
    return uniform_filter1d(oh, size=size, axis=1, mode="nearest").argmax(0)


class Cancelled(Exception):
    pass


class Project:
    def __init__(self, models_factory, work_dir):
        self.models_factory = models_factory
        self._models = None
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.lock = threading.RLock()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.path = None
        self.x = None
        self.chunks = []
        self.chars = []
        self.edits = []
        self.next_id = 1
        self.analysed = False
        self.rendered = False
        self.render_version = 0
        self.null_db = None
        self.null_db_24 = None
        self.est = None
        self.settings = {"voices": 2, "bg": "natural"}
        self.analysed_voices = None
        self.lane_energy = None

    @property
    def models(self):
        if self._models is None:
            self._models = self.models_factory()
        return self._models

    def info(self):
        if self.x is None:
            return None
        return {"path": self.path, "name": os.path.basename(self.path), "sr": self.sr,
                "channels": self.C, "seconds": self.N / self.sr, "time_ref": self.time_ref,
                "start_seconds": (self.time_ref or 0) / self.sr}

    # ------------------------------------------------------------------ 1. load
    def load(self, path):
        x, sr, tref = audio_io.load(path)
        if len(x) < sr * 0.5:
            raise ValueError("This file is shorter than half a second.")
        with self.lock:
            self.reset()
            self.path = path
            self.time_ref = tref
            if sr % 100:
                x = resample(x, sr, 48000)
                if tref:
                    tref = int(round(tref * 48000 / sr))
                    self.time_ref = tref
                sr = 48000
            self.x = np.ascontiguousarray(x, np.float32)
            self.sr, self.N, self.C = sr, len(x), x.shape[1]
            self.hop = sr // FPS
            self.nfft = 4 * self.hop
            self.pad = (self.nfft - self.hop) // 2
            self.nF = int(math.ceil((self.N + self.pad) / self.hop)) + 1
            self.L16 = self.nF * HOP16
            mono = self.x.mean(1)
            m16 = resample(mono, sr, SR16)
            self.mono16 = np.zeros(self.L16, np.float32)
            k = min(len(m16), self.L16)
            self.mono16[:k] = m16[:k]
            self.win = np.hanning(self.nfft + 1)[:-1].astype(np.float32)
            self.win16 = np.hanning(NFFT16 + 1)[:-1].astype(np.float32)
            self._vad()
            self.project_dir = os.path.join(self.work_dir, "p%d" % int(time.time() * 1000))
            os.makedirs(self.project_dir, exist_ok=True)
            # a 24-bit copy of the original to listen to in the app
            self.orig_preview = os.path.join(self.project_dir, "original.wav")
            w = audio_io.WavWriter(self.orig_preview, sr, self.C, self.N)
            for s in range(0, self.N, sr * 30):
                w.write_at(s, self.x[s:s + sr * 30])
            w.close()
        return self.info()

    def _vad(self):
        e = frame_energy(self.mono16, self.nF, HOP16)
        d = db(e)
        live = d[d > -100]
        floor = np.percentile(live, 10) if len(live) else -100
        self.floor_db = float(floor)
        thr = max(floor + 9.0, -66.0)
        sp = d > thr
        sp = drop_short(sp, 5)
        sp = fill_gaps(sp, 25)
        self.speech_raw = sp
        self.mix_e = e

    def peaks(self, per_second=100):
        """min/max of the original for drawing, 100 columns per second, int8."""
        mono = self.x.mean(1)
        hop = self.sr // per_second
        n = int(math.ceil(self.N / hop))
        y = np.zeros(n * hop, np.float32)
        y[:self.N] = mono
        y = y.reshape(n, hop)
        mx = np.clip(np.round(y.max(1) * 127), -127, 127).astype(np.int8)
        mn = np.clip(np.round(y.min(1) * 127), -127, 127).astype(np.int8)
        return mn, mx

    # ------------------------------------------------------------------ 2. analyse
    def analyse(self, progress=lambda *a: None, cancel=lambda: False):
        voices = int(self.settings["voices"])
        M = self.models
        L = M.chunk_len16(voices)
        ov = int(OVERLAP_S * SR16)
        step = L - ov
        # stretches to separate: talking plus 0.3 s either side
        reg = dilate(self.speech_raw, 30, 30)
        reg = fill_gaps(reg, 50)
        starts = []
        for a, b in runs(reg):
            a16, b16 = a * HOP16, b * HOP16
            if b16 - a16 <= L:
                s = max(0, min(a16 - (L - (b16 - a16)) // 2, self.L16 - L))
                starts.append((s, a16, b16))
            else:
                s = a16
                while True:
                    starts.append((s, a16, b16))
                    if s + L >= b16:
                        break
                    s = min(s + step, b16 - L)
        if not starts:
            raise ValueError("No dialogue was found in this file.")
        chunks = []
        t0 = time.time()
        wst = np.arange(0, L - WIN16 + 1, WSTEP16)
        for ci, (s, a16, b16) in enumerate(starts):
            if cancel():
                raise Cancelled()
            seg = np.zeros(L, np.float32)
            k = max(0, min(L, self.L16 - s))
            seg[:k] = self.mono16[s:s + k]
            rms = float(np.sqrt((seg ** 2).mean()))
            g = float(np.clip(0.05 / max(rms, 1e-6), 0.05, 2000))
            if hasattr(M, "locate"):
                M.locate(s)
            out = M.separate(seg * g, voices) / g
            wins = out[:, wst[:, None] + np.arange(WIN16)[None, :]]          # (v, nW, WIN)
            wdb = db((wins.astype(np.float64) ** 2).mean(2))
            emb = M.embed(wins.reshape(-1, WIN16)).reshape(out.shape[0], len(wst), -1)
            # crossfade weight: ramps where pieces overlap, flat elsewhere
            w = np.ones(L, np.float32)
            r = np.linspace(0.001, 1, ov, dtype=np.float32)
            if s > a16:
                w[:ov] = r
            if s + L < b16:
                w[-ov:] = r[::-1]
            chunks.append({"start": s, "out": out.astype(np.float32), "emb": emb.astype(np.float32),
                           "wdb": wdb, "wst": wst, "w": w})
            done = ci + 1
            el = time.time() - t0
            progress(5 + 90 * done / len(starts), "Separating voices (%d of %d)" % (done, len(starts)),
                     el / done * (len(starts) - done))
        with self.lock:
            self.chunks = chunks
            self.analysed = True
            self.analysed_voices = voices
            self.rendered = False
        progress(100, "Analysed", 0)

    def _windows(self, rel_db=30.0):
        """All voice-identity windows that carry real voice: (embeddings, chunk, output, window)."""
        E, idx = [], []
        for ci, c in enumerate(self.chunks):
            top = c["wdb"].max()
            ok = (c["wdb"] > top - rel_db) & (c["wdb"] > self.floor_db + 3)
            ii, ww = np.nonzero(ok)
            E.append(c["emb"][ii, ww])
            idx += [(ci, i, w) for i, w in zip(ii, ww)]
        if not idx:
            return np.zeros((0, 1), np.float32), []
        return np.concatenate(E), idx

    # ------------------------------------------------------------------ 3. characters
    def find_characters(self, k=None):
        E, idx = self._windows(20.0)
        if len(E) < 4:
            raise ValueError("Not enough clear dialogue to tell the voices apart.")
        rng = np.random.default_rng(0)
        sub = E if len(E) <= 2500 else E[rng.choice(len(E), 2500, replace=False)]
        Z = linkage(sub, method="average", metric="cosine")
        if k:
            lab = fcluster(Z, t=int(k), criterion="maxclust")
        else:
            lab = fcluster(Z, t=0.62, criterion="distance")
        cents = []
        for c in np.unique(lab):
            m = lab == c
            if k or m.sum() >= max(3, 0.03 * len(sub)):
                cents.append(sub[m].mean(0))
        if len(cents) < 2 and len(sub) >= 2:
            lab = fcluster(Z, t=2, criterion="maxclust")
            cents = [sub[lab == c].mean(0) for c in np.unique(lab)]
        C = np.stack(cents)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        for _ in range(6):                                  # refine on every window
            a = (E @ C.T).argmax(1)
            newC = []
            for j in range(len(C)):
                if (a == j).any():
                    v = E[a == j].mean(0)
                    newC.append(v / (np.linalg.norm(v) + 1e-9))
            C = np.stack(newC)
        a = (E @ C.T).argmax(1)
        talk = np.bincount(a, minlength=len(C)).astype(float) * WSTEP16 / SR16
        order = np.argsort(-talk)
        with self.lock:
            self.chars = []
            self.edits = []
            for n, j in enumerate(order):
                self._new_char("Character %d" % (n + 1), C[j], auto=True)
            self.rendered = False
        return self.chars

    def _new_char(self, name, centroid, auto=False):
        cid = self.next_id
        self.next_id += 1
        used = {c["color"] for c in self.chars}
        color = next((c for c in COLORS if c not in used), COLORS[cid % len(COLORS)])
        ch = {"id": cid, "name": name, "color": color, "c": centroid.astype(np.float32), "sample": None}
        self.chars.append(ch)
        return ch

    def voice_from_selection(self, t0, t1):
        """Voice-identity of the clearest voice inside a selection."""
        a, b = int(t0 * SR16), int(t1 * SR16)
        if b - a < SR16 * 0.4:
            raise ValueError("Select at least half a second of this character talking on their own.")
        best = None
        for c in self.chunks:
            s, L = c["start"], c["out"].shape[1]
            lo, hi = max(a, s), min(b, s + L)
            if hi - lo < SR16 * 0.4:
                continue
            seg = c["out"][:, lo - s:hi - s]
            e = (seg ** 2).sum(1)
            i = int(e.argmax())
            if best is None or hi - lo > best[0]:
                best = (hi - lo, seg[i])
        sig = best[1] if best is not None else self.mono16[a:b]
        if len(sig) < WIN16:
            sig = np.tile(sig, int(np.ceil(WIN16 / len(sig))))[:WIN16]
        st = np.arange(0, len(sig) - WIN16 + 1, WSTEP16)
        wins = sig[st[:, None] + np.arange(WIN16)[None, :]]
        e = db((wins.astype(np.float64) ** 2).mean(1))
        wins = wins[e > e.max() - 15]
        v = self.models.embed(wins.astype(np.float32)).mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def add_character(self, name, t0, t1):
        v = self.voice_from_selection(t0, t1)
        with self.lock:
            ch = self._new_char(name or "Character %d" % (len(self.chars) + 1), v)
            ch["sample"] = [t0, t1]
            self.rendered = False
        return ch

    def set_sample(self, cid, t0, t1):
        v = self.voice_from_selection(t0, t1)
        with self.lock:
            ch = self.char(cid)
            ch["c"] = v
            ch["sample"] = [t0, t1]
            self.rendered = False

    def char(self, cid):
        for c in self.chars:
            if c["id"] == int(cid):
                return c
        raise KeyError("No such character")

    def rename(self, cid, name):
        with self.lock:
            self.char(cid)["name"] = (name or "").strip()[:60] or "Character"

    def delete_char(self, cid):
        with self.lock:
            self.chars = [c for c in self.chars if c["id"] != int(cid)]
            self._clean_edits()
            self.rendered = False

    def merge(self, src, dst):
        with self.lock:
            a, b = self.char(src), self.char(dst)
            v = a["c"] + b["c"]
            b["c"] = v / (np.linalg.norm(v) + 1e-9)
            b.setdefault("extra", []).append(a["c"])
            b["extra"] += a.get("extra", [])
            self.chars = [c for c in self.chars if c["id"] != a["id"]]
            for e in self.edits:
                e["chars"] = [dst if c == src else c for c in e["chars"]]
            self._clean_edits()
            self.rendered = False

    def _clean_edits(self):
        ids = {c["id"] for c in self.chars}
        keep = []
        for e in self.edits:
            e["chars"] = sorted(set(c for c in e["chars"] if c in ids))
            if e["kind"] == "residual" or e["chars"]:
                keep.append(e)
        self.edits = keep

    # ------------------------------------------------------------------ 4. who is talking when
    def _scores(self, emb):
        """Similarity of embeddings to each character (best of the character's voice prints)."""
        out = []
        for ch in self.chars:
            vs = [ch["c"]] + ch.get("extra", [])
            out.append(np.max(np.stack([emb @ v for v in vs]), 0))
        return np.stack(out, -1)

    def assign(self, progress=lambda *a: None):
        K = len(self.chars)
        if K < 1:
            raise ValueError("Add at least one character first.")
        est = np.zeros((K, self.L16), np.float32)
        den = np.zeros(self.L16, np.float32)
        for n, c in enumerate(self.chunks):
            s = c["start"]
            out = c["out"]
            L = out.shape[1]
            e = min(s + L, self.L16)
            k = e - s
            w = c["w"][:k]
            sc = self._scores(c["emb"])                                  # (v, nW, K)
            top = c["wdb"].max()
            valid = (c["wdb"] > top - 30) & (c["wdb"] > self.floor_db + 3)
            centers = (c["wst"] + WIN16 / 2) / HOP16
            nfr = L // HOP16
            fr = np.arange(nfr) + 0.5
            nv = out.shape[0]
            curves = np.zeros((nv, K, nfr), np.float32)
            for i in range(nv):
                v = valid[i]
                if v.sum() == 0:
                    curves[i] = sc[i].mean(0)[:, None]
                else:
                    curves[i] = np.stack([np.interp(fr, centers[v], sc[i, v, j]) for j in range(K)])
            labs = curves.argmax(1)                                          # (v, nfr)
            if nv > 1 and K > 1:
                # two voices the network pulled apart at the same moment are two different people
                fe = db(np.stack([frame_energy(out[i], nfr, HOP16) for i in range(nv)]))
                act = fe > max(fe.max() - 30, self.floor_db + 3)
                both = act.sum(0) >= 2
                if both.any():
                    import itertools
                    perms = np.array(list(itertools.permutations(range(K), min(nv, K))))
                    tot = np.zeros((len(perms), nfr), np.float32)
                    for i in range(perms.shape[1]):
                        tot += np.where(act[i][None], curves[i][perms[:, i]], 0)
                    best = perms[tot.argmax(0)]                              # (nfr, v)
                    for i in range(perms.shape[1]):
                        labs[i, both] = best[both, i]
            for i in range(nv):
                lab = majority(labs[i], K, 15)
                oh = np.zeros((K, nfr), np.float32)
                oh[lab, np.arange(nfr)] = 1
                oh = uniform_filter1d(oh, 3, axis=1, mode="nearest")
                A = np.repeat(oh, HOP16, axis=1)[:, :k]
                est[:, s:e] += (w * out[i, :k])[None, :] * A
            den[s:e] += w
            if n % 20 == 0:
                progress(5 + 25 * n / max(1, len(self.chunks)), "Matching voices to characters", None)
        est /= np.maximum(den, 1e-6)[None, :]
        self.est = est
        self.E = np.stack([frame_energy(est[j], self.nF, HOP16) for j in range(K)])
        self.char_ids = [c["id"] for c in self.chars]

    def decide(self):
        """Frame by frame: who is talking, and where voices overlap."""
        K, nF = len(self.chars), self.nF
        Es = uniform_filter1d(self.E, 5, axis=1, mode="nearest")
        top = Es.max(0)
        floor = 10 ** ((self.floor_db + 6) / 10)
        active = (Es > top[None, :] * 10 ** (-14 / 10)) & (Es > floor)
        speech = self.speech_raw | (active.any(0) & (top > floor * 2))
        o_raw = (active.sum(0) >= 2) & speech
        dom = majority(Es.argmax(0), K, 9) if K > 1 else np.zeros(nF, int)
        sw = dilate(speech, 6, 18).astype(np.float32)
        o = dilate(o_raw, 8, 8).astype(np.float32)
        allow = np.ones((K, nF), bool)
        pos = {c["id"]: j for j, c in enumerate(self.chars)}
        for e in self.edits:
            a, b = max(0, int(e["t0"] * FPS)), min(nF, int(math.ceil(e["t1"] * FPS)))
            if b <= a:
                continue
            if e["kind"] == "residual":
                sw[a:b] = 0
            elif e["kind"] == "assign":
                j = pos[e["chars"][0]]
                sw[a:b] = 1
                o[a:b] = 0
                dom[a:b] = j
            elif e["kind"] == "overlap":
                js = [pos[c] for c in e["chars"]]
                sw[a:b] = 1
                o[a:b] = 1 if len(js) > 1 else 0
                allow[:, a:b] = False
                allow[js, a:b] = True
                if len(js) == 1:
                    dom[a:b] = js[0]
                else:
                    sub = Es[js][:, a:b]
                    dom[a:b] = np.array(js)[sub.argmax(0)]
        hard = np.zeros((K, nF), np.float32)
        hard[dom, np.arange(nF)] = 1
        self.hard = uniform_filter1d(hard, 3, axis=1, mode="nearest")      # 30 ms crossfades
        self.sw = np.clip(ramp(sw, 5), 0, 1)
        self.o = np.clip(ramp(o, 6), 0, 1)
        self.allow = allow
        self.o_raw = o_raw

    # ------------------------------------------------------------------ 5. build the stems
    def _masks(self, j0, j1, X):
        """Share-out of each frequency of frames j0..j1 between characters and the residual.
        X: (C, B, bins) spectrum of the original. Returns (K+1, B, bins), summing to 1."""
        K = len(self.chars)
        B = j1 - j0
        nb = self.nfft // 2 + 1
        valid_hz = self.models.model_sr(self.analysed_voices) / 2
        f16 = np.arange(NFFT16 // 2 + 1) * (SR16 / NFFT16)
        # power of each character's separated voice, 16 kHz analysis aligned frame for frame
        idx = (np.arange(j0, j1) * HOP16)[:, None] + np.arange(NFFT16)[None, :]
        estp = self._estp
        P = np.abs(np.fft.rfft(estp[:, idx] * self.win16, axis=-1)) ** 2               # (K,B,321)
        P *= self.allow[:, j0:j1, None]
        hard = self.hard[:, j0:j1]
        S = P.sum(0)
        tiny = 1e-12
        soft = np.where(S[None] > tiny, P / np.maximum(S, tiny)[None], hard[:, :, None])
        # above what the network hears: follow the share of the top of its range
        band = (f16 >= valid_hz * 0.4) & (f16 < valid_hz * 0.95)
        W = (np.abs(X[:, :, :len(f16)]) ** 2).sum(0)[:, band] + 1e-12                 # (B, nbnd)
        hb = (soft[:, :, band] * W[None]).sum(-1) / W.sum(-1)[None]                     # (K,B)
        fn = np.arange(nb) * (self.sr / self.nfft)
        soft_n = np.empty((K, B, nb), np.float32)
        m = min(nb, len(f16))
        soft_n[:, :, :m] = soft[:, :, :m]
        soft_n[:, :, m:] = hb[:, :, None]
        blend = np.clip((fn - valid_hz * 0.85) / (valid_hz * 0.15), 0, 1)               # (nb,)
        soft_n = soft_n * (1 - blend) + hb[:, :, None] * blend
        o = self.o[j0:j1][None, :, None]
        voice = (1 - o) * hard[:, :, None] + o * soft_n
        if self.settings.get("bg") == "clean":
            R = np.abs(np.fft.rfft(self._resp[idx] * self.win16, axis=-1)) ** 2
            nz = R / (S + R + tiny)
            nzh = (nz[:, band] * W).sum(-1) / W.sum(-1)
            nz_n = np.empty((B, nb), np.float32)
            nz_n[:, :m] = nz[:, :m]
            nz_n[:, m:] = nzh[:, None]
            nz_n = nz_n * (1 - blend) + nzh[:, None] * blend
            voice = voice * (1 - np.clip(nz_n, 0, 0.97))[None]
        Mk = self.sw[j0:j1][None, :, None] * voice
        Mres = np.clip(1 - Mk.sum(0), 0, 1)
        return np.concatenate([Mk, Mres[None]], 0).astype(np.float32)

    def stem_names(self):
        return [c["name"] for c in self.chars] + ["Residual"]

    def stem_paths(self):
        return [os.path.join(self.project_dir, "stem_%d.wav" % i) for i in range(len(self.chars) + 1)]

    def _prepare_render(self):
        K = len(self.chars)
        L = self.nF * HOP16 + NFFT16
        self._estp = np.zeros((K, L), np.float32)
        self._estp[:, PAD16:PAD16 + self.L16] = self.est
        if self.settings.get("bg") == "clean":
            self._resp = np.zeros(L, np.float32)
            self._resp[PAD16:PAD16 + self.L16] = self.mono16 - self.est.sum(0)
        Lp = self.nF * self.hop + self.nfft
        self._xp = np.zeros((self.C, Lp), np.float32)
        self._xp[:, self.pad:self.pad + self.N] = self.x.T

    def render(self, progress=lambda *a: None, frames=None, out_paths=None, fmt="pcm24", cancel=lambda: False):
        """Write the stems. frames=(j0, j1) re-renders just that stretch in place."""
        K = len(self.chars)
        self._prepare_render()
        hop, nfft, pad = self.hop, self.nfft, self.pad
        paths = out_paths or self.stem_paths()
        patch = frames is not None and out_paths is None and self.rendered
        if not patch:
            frames = (0, self.nF)
        names = self.stem_names()
        writers = [audio_io.WavWriter(p, self.sr, self.C, self.N, fmt=fmt, time_ref=self.time_ref,
                                      description="Un-tie stem: " + names[i], patch=patch)
                   for i, p in enumerate(paths)]
        if self.lane_energy is None or self.lane_energy.shape[0] != K + 1 or not patch:
            self.lane_energy = np.zeros((K + 1, self.nF), np.float32)
        null_max, null24_max = 0.0, 0.0
        win = self.win
        w2 = win ** 2
        f0, f1 = max(0, frames[0]), min(self.nF, frames[1])
        t0 = time.time()
        try:
            for b0 in range(f0, f1, BLOCK):
                if cancel():
                    raise Cancelled()
                b1 = min(f1, b0 + BLOCK)
                a0 = max(0, b0 - 3)
                nfr = b1 - a0
                idx = (np.arange(a0, b1) * hop)[:, None] + np.arange(nfft)[None, :]
                X = np.fft.rfft(self._xp[:, idx] * win, axis=-1)                    # (C,nfr,bins)
                M = self._masks(a0, b1, X)                                          # (K+1,nfr,bins)
                self.lane_energy[:, b0:b1] = ((M[:, b0 - a0:] ** 2)[:, None]
                                              * (np.abs(X[:, b0 - a0:]) ** 2)[None]).sum((1, 3))
                # overlap-add weights for exactly these frames
                ws = np.zeros((nfr + 3) * hop, np.float32)
                for q in range(nfr):
                    ws[q * hop:q * hop + nfft] += w2
                o_lo, o_hi = (b0 - a0) * hop, (b1 - a0) * hop
                ws = ws[o_lo:o_hi]
                s_orig = b0 * hop - pad                       # first original sample of this block
                total = None
                for s in range(K + 1):
                    y = np.fft.irfft(X * M[s][None], n=nfft, axis=-1).astype(np.float32) * win
                    buf = np.zeros((self.C, nfr + 3, hop), np.float32)
                    yy = y.reshape(self.C, nfr, 4, hop)
                    for q in range(4):
                        buf[:, q:q + nfr] += yy[:, :, q]
                    out = buf.reshape(self.C, -1)[:, o_lo:o_hi] / np.maximum(ws, 1e-9)
                    lo = max(0, -s_orig)
                    hi = min(out.shape[1], self.N - s_orig)
                    if hi <= lo:
                        continue
                    seg = out[:, lo:hi].T
                    writers[s].write_at(s_orig + lo, seg)
                    total = seg.astype(np.float64) if total is None else total + seg
                    if s == 0:
                        tq = audio_io.quantize24(seg)
                    else:
                        tq = tq + audio_io.quantize24(seg)
                if total is not None:
                    ref = self.x[s_orig + lo:s_orig + hi].astype(np.float64)
                    null_max = max(null_max, float(np.abs(total - ref).max(initial=0)))
                    null24_max = max(null24_max, float(np.abs(tq - audio_io.quantize24(ref)).max(initial=0)))
                done = (b1 - f0) / max(1, f1 - f0)
                el = time.time() - t0
                progress(30 + 70 * done, "Building stems", el / max(done, 1e-3) * (1 - done))
        finally:
            for w in writers:
                w.close()
        if out_paths is None:
            if not patch:
                self.null_db = 20 * math.log10(max(null_max, 1e-12))
                self.null_db_24 = 20 * math.log10(max(null24_max, 1e-12))
            else:
                self.null_db = max(self.null_db or -240, 20 * math.log10(max(null_max, 1e-12)))
                self.null_db_24 = max(self.null_db_24 or -240, 20 * math.log10(max(null24_max, 1e-12)))
            self.rendered = True
            self.render_version += 1
        return 20 * math.log10(max(null_max, 1e-12))

    def untie(self, progress=lambda *a: None, cancel=lambda: False):
        if not self.analysed:
            raise ValueError("Analyse the file first.")
        if len(self.chars) < 1:
            raise ValueError("Add at least one character first.")
        with self.lock:
            self.rendered = False
            self.assign(progress)
            self.decide()
            self.render(progress, cancel=cancel)
        progress(100, "Done", 0)

    # ------------------------------------------------------------------ 6. fixes
    def add_edit(self, kind, t0, t1, chars, progress=lambda *a: None):
        t0, t1 = sorted((float(t0), float(t1)))
        if t1 - t0 < 0.01:
            raise ValueError("Select a stretch of the timeline first.")
        if kind not in ("assign", "overlap", "residual"):
            raise ValueError("unknown fix")
        chars = [int(c) for c in chars]
        if kind == "assign" and len(chars) != 1:
            raise ValueError("Choose one character.")
        if kind == "overlap" and not chars:
            raise ValueError("Tick the characters talking here.")
        with self.lock:
            self.edits.append({"kind": kind, "t0": t0, "t1": t1, "chars": chars})
            self._rerender(t0, t1, progress)

    def remove_edit(self, i, progress=lambda *a: None):
        with self.lock:
            e = self.edits.pop(int(i))
            self._rerender(e["t0"], e["t1"], progress)

    def _rerender(self, t0, t1, progress):
        if not self.rendered or self.est is None:
            return
        self.decide()
        self.render(progress, frames=(int(t0 * FPS) - 60, int(math.ceil(t1 * FPS)) + 60))

    # ------------------------------------------------------------------ 7. export
    def export(self, folder, fmt="pcm24", residual=True, progress=lambda *a: None,
               wavs=True, aaf=False, start_tc=None, fps=25.0, guide=True):
        """Writes WAV stems and/or a Pro Tools AAF (all stems as tracks, audio embedded)."""
        if not self.rendered:
            raise ValueError("Un-tie the file first.")
        if not wavs and not aaf:
            raise ValueError("Choose WAV stems, an AAF, or both.")
        base = os.path.splitext(os.path.basename(self.path))[0]
        names = self.stem_names()
        safe = []
        for n in names:
            s = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in n).strip() or "Character"
            while s in safe:
                s += "_2"
            safe.append(s)
        outs = []
        if wavs:
            outs = [os.path.join(folder, "%s_UNTIE_%s.wav" % (base, s)) for s in safe]
            wav_share = 50 if aaf else 100
            prog = (lambda p, m, e: progress(p * wav_share / 100, "Exporting stems", None))
            if fmt == "float32":
                self.render(prog, out_paths=outs, fmt="float32")
            else:
                import shutil
                for i, (src, dst) in enumerate(zip(self.stem_paths(), outs)):
                    shutil.copyfile(src, dst)
                    prog(100 * (i + 1) / len(outs), "", None)
            if not residual:
                os.remove(outs[-1])
                outs = outs[:-1]
        if aaf:
            from . import aaf_export
            tc, lead = aaf_export.default_start_tc(self.time_ref, self.sr, fps)
            if start_tc and start_tc.strip() != tc:
                tc, lead = start_tc.strip(), 0          # the user typed their own start
            tracks = [(n, p) for n, p in zip(names, self.stem_paths())]
            if not residual:
                tracks = tracks[:-1]
            if guide:
                tracks.append(("Original mix", self.orig_preview))
            path = os.path.join(folder, "%s_UNTIE.aaf" % base)
            off = 50 if wavs else 0
            aaf_export.write_aaf(path, base + " UNTIE", tracks, self.sr, tc, fps,
                                 lambda p, m, e: progress(off + p * (100 - off) / 100, m, None), lead=lead)
            outs.append(path)
            self.aaf_tc = tc
        rep = os.path.join(folder, "%s_UNTIE_report.txt" % base)
        with open(rep, "w") as f:
            f.write("Un-tie by Sound Brewery\nSource: %s\n" % self.path)
            f.write("Sample rate: %d Hz, %d channel(s), %.3f s\n" % (self.sr, self.C, self.N / self.sr))
            if self.time_ref is not None:
                f.write("Timecode stamp kept from the source (%d samples since midnight)\n" % self.time_ref)
            if aaf:
                f.write("AAF starts at %s (%s fps); stereo stems are L/R mono track pairs\n" % (tc, fps))
            f.write("Null test (all stems + residual vs original): %.1f dB\n" % (self.null_db or -999))
            f.write("Fixes applied: %d\n\nFiles:\n" % len(self.edits))
            for p in outs:
                f.write("  " + os.path.basename(p) + "\n")
        return outs

    # ------------------------------------------------------------------ for the timeline
    def overlaps(self):
        if not self.rendered:
            return []
        return [(a / FPS, b / FPS) for a, b in runs(self.o > 0.5)]

    def lanes(self):
        """Loudness of each stem, 100 per second, as 0..255 (-80..0 dB)."""
        if self.lane_energy is None:
            return []
        # frame j is centred at (j + 0.5) / 100 s; levels relative to the loudest moment
        d = db(self.lane_energy)
        d = d - d.max()
        return np.clip((d + 72) / 72 * 255, 0, 255).astype(np.uint8)
