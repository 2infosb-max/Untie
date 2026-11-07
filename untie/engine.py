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
OVERLAP_S = 1.0
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
        self.settings = {"voices": 2, "bg": "natural", "thorough": False, "three": False}
        self.mix_win = None
        self.separated_share = None
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
        talk = fill_gaps(dilate(self.speech_raw, 30, 30), 50)
        if not talk.any():
            raise ValueError("No dialogue was found in this file.")
        # quick pass: a voice print of the original mix every 0.25 s
        progress(2, "Listening to the voices", None)
        self._mix_pass(M, talk, progress, cancel)
        if self.settings.get("thorough"):
            reg = talk
        else:
            # only where it isn't clearly one known voice does the slow separation run
            reg = self._unclear_regions(talk)
        self.separated_share = float(reg.sum() / max(1, talk.sum()))
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
        chunks = []
        t0 = time.time()
        for ci, (s, a16, b16) in enumerate(starts):
            if cancel():
                raise Cancelled()
            # crossfade weight: ramps where pieces overlap, flat elsewhere
            w = np.ones(L, np.float32)
            r = np.linspace(0.001, 1, ov, dtype=np.float32)
            if s > a16:
                w[:ov] = r
            if s + L < b16:
                w[-ov:] = r[::-1]
            chunks.append(self._separate_piece(M, s, L, voices, w))
            done = ci + 1
            el = time.time() - t0
            progress(20 + 75 * done / len(starts), "Separating overlapping voices (%d of %d)" % (done, len(starts)),
                     el / done * (len(starts) - done) * 1.15)
        # second pass: every overlap again, centred in its own piece so the network hears both
        # voices with the most context and no third person
        regions = self._overlap_regions(chunks) if self.REFINE_OVERLAPS else []
        per = (time.time() - t0) / max(1, len(starts))
        for ri, (a16, b16) in enumerate(regions):
            if cancel():
                raise Cancelled()
            s = int(max(0, min((a16 + b16) // 2 - L // 2, self.L16 - L)))
            w = np.full(L, 0.001, np.float32)
            lo, hi = max(0, a16 - s - 4800), min(L, b16 - s + 4800)
            w[lo:hi] = self.REFINE_WEIGHT
            w = np.maximum(uniform_filter1d(w, 3200, mode="nearest"), 0.001).astype(np.float32)
            chunks.append(self._separate_piece(M, s, L, voices, w))
            progress(80 + 15 * (ri + 1) / len(regions), "Refining overlaps (%d of %d)" % (ri + 1, len(regions)),
                     per * (len(regions) - ri - 1))
        with self.lock:
            self.chunks = chunks
            self.analysed = True
            self.analysed_voices = voices
            self.rendered = False
        progress(100, "Analysed", 0)

    UNCLEAR_TOP = 0.5          # mix voice print less like any character than this = unclear
    UNCLEAR_MARGIN = 0.2       # ...or this close to two characters = unclear

    def _mix_pass(self, M, talk, progress, cancel):
        st = []
        for a, b in runs(talk):
            a16, b16 = a * HOP16, b * HOP16
            st += list(range(max(0, a16 - WIN16 // 2), max(a16 + 1, min(self.L16 - WIN16, b16 - WIN16 // 2)), WSTEP16))
        st = np.array(sorted(set(st)), int)
        W = self.mono16[st[:, None] + np.arange(WIN16)[None, :]].astype(np.float32)
        E = np.zeros((len(st), 0), np.float32)
        parts = []
        for i in range(0, len(st), 256):
            if cancel():
                raise Cancelled()
            parts.append(M.embed(W[i:i + 256]))
            progress(2 + 16 * min(1, (i + 256) / len(st)), "Listening to the voices", None)
        E = np.concatenate(parts) if parts else E
        self.mix_win = {"st": st, "emb": E.astype(np.float32),
                        "db": db((W.astype(np.float64) ** 2).mean(1))}

    def _mix_segments(self):
        """~1.5 s stretches of the mix whose voice prints agree with each other: one person alone."""
        mw = getattr(self, "mix_win", None)
        if mw is None or len(mw["st"]) == 0:
            return np.zeros((0, 1), np.float32), np.zeros(0)
        st, E, d = mw["st"], mw["emb"], mw["db"]
        ok = d > self.floor_db + 6
        E_, Wt = [], []
        brk = np.concatenate([[0], np.nonzero(np.diff(st) != WSTEP16)[0] + 1, [len(st)]])
        for a, b in zip(brk[:-1], brk[1:]):
            for q in range(a, b, 3):
                r = np.arange(q, min(b, q + 6))
                r = r[ok[r]]
                if len(r) < 4:
                    continue
                v = E[r].mean(0)
                v /= np.linalg.norm(v) + 1e-9
                if (E[r] @ v).mean() > 0.6:
                    E_.append(v)
                    Wt.append(len(r))
        if not E_:
            return np.zeros((0, 1), np.float32), np.zeros(0)
        return np.stack(E_), np.array(Wt, float)

    def _unclear_regions(self, talk):
        """Talking frames whose mix voice print isn't clearly one of the voices in the file."""
        mw = self.mix_win
        E, _ = self._mix_segments()
        if len(E) < 4:
            return talk
        Z = linkage(E, method="average", metric="cosine")
        lab = fcluster(Z, t=self.CLUSTER_DIST * 0.9, criterion="distance")   # err towards more voices
        C = np.stack([E[lab == c].mean(0) for c in np.unique(lab)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        S = mw["emb"] @ C.T
        srt = np.sort(S, 1)
        top = srt[:, -1]
        margin = srt[:, -1] - srt[:, -2] if S.shape[1] > 1 else np.ones(len(S))
        loud = mw["db"] > self.floor_db + 6
        flag = loud & ((top < self.UNCLEAR_TOP) | (margin < self.UNCLEAR_MARGIN))
        mark = np.zeros(self.nF, bool)
        for a16 in mw["st"][flag]:
            f = a16 // HOP16
            mark[max(0, f - 25):min(self.nF, f + 125)] = True                 # the window +- 0.25 s
        return fill_gaps(mark & talk, 100) & talk

    REFINE_WEIGHT = 4.0
    REFINE_OVERLAPS = False    # tested: no gain on the reference scene for +50% time

    def _separate_piece(self, M, s, L, voices, w):
        wst = np.arange(0, L - WIN16 + 1, WSTEP16)
        seg = np.zeros(L, np.float32)
        k = max(0, min(L, self.L16 - s))
        seg[:k] = self.mono16[s:s + k]
        rms = float(np.sqrt((seg ** 2).mean()))
        g = float(np.clip(0.05 / max(rms, 1e-6), 0.05, 2000))
        if hasattr(M, "locate"):
            M.locate(s)
        out = M.separate(seg * g, voices) / g
        out = self._fit_to_mix(out, seg)
        if self.settings.get("three", False):
            out = self._resplit(M, out, wst, s)
        wins = out[:, wst[:, None] + np.arange(WIN16)[None, :]]          # (v, nW, WIN)
        wdb = db((wins.astype(np.float64) ** 2).mean(2))
        emb = M.embed(wins.reshape(-1, WIN16)).reshape(out.shape[0], len(wst), -1)
        memb, solo = self._mix_info(seg, wdb, wst, M)
        return {"start": s, "out": out.astype(np.float32), "emb": emb.astype(np.float32),
                "wdb": wdb, "wst": wst, "w": w, "memb": memb, "solo": solo}

    def _win_info(self, M, x, wst):
        wins = x[:, wst[:, None] + np.arange(WIN16)[None, :]]
        wdb = db((wins.astype(np.float64) ** 2).mean(2))
        emb = M.embed(wins.reshape(-1, WIN16).astype(np.float32)).reshape(x.shape[0], len(wst), -1)
        return wdb, emb

    def _two_people(self, wdb, emb, top):
        """Per window: are both voices really talking, at similar level, with different voice prints?"""
        live = (wdb > top - 30) & (wdb > self.floor_db + 3)
        close = np.abs(wdb[0] - wdb[1]) < self.SOLO_DB
        cos = np.sum(emb[0] * emb[1], -1)
        return live.all(0) & close & (cos < self.DIFFERENT_COS)

    def _resplit(self, M, out, wst, s):
        """Three people at once: the network only splits in two, so one of its two voices still holds
        two people. Split each voice again; where the halves are clearly two different people, keep
        them apart (up to 4 voices per moment). The halves always add back up to the voice they came from."""
        if out.shape[0] != 2:
            return out
        wdb, emb = self._win_info(M, out, wst)
        top = wdb.max()
        if not self._two_people(wdb, emb, top).any():
            return out                                         # no overlap here at all
        L = out.shape[1]
        centers = wst + WIN16 / 2
        res = np.zeros((4, L), np.float32)
        changed = False
        for i in range(2):
            par = out[i]
            rms = float(np.sqrt((par ** 2).mean()))
            if rms < 1e-6:
                res[2 * i] = par
                continue
            g = float(np.clip(0.05 / rms, 0.05, 2000))
            if hasattr(M, "locate"):
                M.locate(s)
            ch = M.separate(par * g, 2) / g
            ch = self._fit_to_mix(ch, par)
            cdb, cemb = self._win_info(M, ch, wst)
            split = self._two_people(cdb, cemb, top).astype(np.float32)
            if not split.any():
                res[2 * i] = par
                continue
            changed = True
            d = np.clip(np.interp(np.arange(L), centers, uniform_filter1d(split, 3, mode="nearest") * 1.5), 0, 1)
            res[2 * i] = d * ch[0] + (1 - d) * par
            res[2 * i + 1] = d * ch[1]
        return res if changed else out

    def _overlap_regions(self, chunks):
        """Where two different voices talk at once (from the first pass), in 16 kHz samples."""
        mark = np.zeros(self.nF, bool)
        for c in chunks:
            if c["out"].shape[0] < 2:
                continue
            wdb = c["wdb"]
            srt = np.sort(wdb, 0)
            live = (srt[-2] > wdb.max() - 30) & (srt[-2] > self.floor_db + 3)
            close = (srt[-1] - srt[-2]) < self.SOLO_DB
            pair = np.sum(c["emb"][0] * c["emb"][1], -1)
            hit = live & close & (pair < self.DIFFERENT_COS)
            for wi in np.nonzero(hit)[0]:
                f = (c["start"] + c["wst"][wi]) // HOP16
                mark[max(0, f + 35):min(self.nF, f + 65)] = True       # middle 0.3 s of the 1 s window
        mark = drop_short(fill_gaps(mark, 30), 10)
        L = self.models.chunk_len16(self.analysed_voices or int(self.settings["voices"]))
        out = []
        for a, b in runs(mark):
            a16, b16 = a * HOP16, b * HOP16
            if b16 - a16 <= L - 2 * SR16:              # leave at least 1 s of context each side
                out.append((a16, b16))
        return out

    @staticmethod
    def _fit_to_mix(out, seg):
        """The network's voices come out at an arbitrary level and polarity. Scale (and flip) them so
        together they rebuild the mix, in overlapping 1 s pieces so it follows level changes."""
        nv, L = out.shape
        win, hop = 16000, 8000
        res = np.zeros_like(out)
        wsum = np.zeros(L, np.float32)
        h = np.hanning(win).astype(np.float32)
        for a in range(0, max(1, L - win + hop), hop):
            b = min(L, a + win)
            w = h[:b - a] if b - a == win else np.hanning(b - a).astype(np.float32)
            A = out[:, a:b].T.astype(np.float64)
            reg = 1e-6 * (A ** 2).sum() / max(1, nv) + 1e-12
            gains = np.linalg.solve(A.T @ A + reg * np.eye(nv), A.T @ seg[a:b].astype(np.float64))
            res[:, a:b] += (gains[:, None] * out[:, a:b]) * w[None]
            wsum[a:b] += w
        return res / np.maximum(wsum, 1e-3)[None]

    SOLO_DB = 9.0              # one separated voice this much louder than the other = one person talking
    CLUSTER_DIST = 0.55        # voice-print distance that splits two characters (auto count)

    @staticmethod
    def _mix_info(seg, wdb, wst, M):
        """Voice prints of the ORIGINAL mix (cleaner than separated voices) and which 1 s windows
        have just one person talking."""
        mw = seg[wst[:, None] + np.arange(WIN16)[None, :]]
        memb = M.embed(mw.astype(np.float32)).astype(np.float32)
        if wdb.shape[0] > 1:
            srt = np.sort(wdb, 0)
            solo = (srt[-1] - srt[-2]) > Project.SOLO_DB
        else:
            solo = np.ones(wdb.shape[1], bool)
        return memb, solo

    def _segments(self):
        """~2 s stretches of one person talking alone, each with an averaged voice print."""
        E, W = [], []
        for c in self.chunks:
            if "memb" not in c:
                continue
            top = c["wdb"].max()
            ok = c["solo"] & (c["wdb"].max(0) > top - 20) & (c["wdb"].max(0) > self.floor_db + 6)
            for a, b in runs(ok):
                for q in range(a, b, 6):
                    r = slice(q, min(b, q + 6))
                    if r.stop - r.start < 2:
                        continue
                    v = c["memb"][r].mean(0)
                    E.append(v / (np.linalg.norm(v) + 1e-9))
                    W.append(r.stop - r.start)
        if not E:
            return np.zeros((0, 1), np.float32), np.zeros(0)
        return np.stack(E), np.array(W, float)

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
        E, Wt = self._mix_segments()
        if len(E) < 4:
            E, Wt = self._segments()
        if len(E) < 4:                                     # hardly any solo lines: use separated voices
            E, _ = self._windows(20.0)
            Wt = np.ones(len(E))
        if len(E) < 4:
            raise ValueError("Not enough clear dialogue to tell the voices apart.")
        rng = np.random.default_rng(0)
        pick = np.arange(len(E)) if len(E) <= 2500 else rng.choice(len(E), 2500, replace=False)
        sub, sw = E[pick], Wt[pick]
        Z = linkage(sub, method="average", metric="cosine")
        if k:
            lab = fcluster(Z, t=min(int(k), len(sub)), criterion="maxclust")
        else:
            lab = fcluster(Z, t=self.CLUSTER_DIST, criterion="distance")
        cents = []
        for c in np.unique(lab):
            m = lab == c
            if k or (m.sum() >= 2 and sw[m].sum() >= 0.03 * sw.sum()):
                v = (sub[m] * sw[m, None]).sum(0)
                cents.append(v)
        if len(cents) < 2 and len(sub) >= 2 and not k:
            lab = fcluster(Z, t=2, criterion="maxclust")
            cents = [(sub[lab == c] * sw[lab == c, None]).sum(0) for c in np.unique(lab)]
        C = np.stack(cents)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        for _ in range(6):                                  # refine on every segment
            a = (E @ C.T).argmax(1)
            newC = []
            for j in range(len(C)):
                if (a == j).any():
                    v = (E[a == j] * Wt[a == j, None]).sum(0)
                    newC.append(v / (np.linalg.norm(v) + 1e-9))
            C = np.stack(newC)
        a = (E @ C.T).argmax(1)
        talk = np.bincount(a, weights=Wt, minlength=len(C)).astype(float)
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

    DISTINCT = True
    DIFFERENT_COS = 0.4        # voice prints of two separated voices below this = two people
    MASK_POWER = 1.0
    ISLAND_MIX = 0.0          # how much a whole stretch of talking decides its own label

    def _island_scores(self, out, sc, valid, wst, wdb, curves):
        """A separated voice that talks without a break is one person: judge each unbroken stretch
        as a whole (all its windows together), so single unclear words don't flip to someone else."""
        nv, K, nfr = curves.shape
        res = curves.copy()
        wa = wst // HOP16
        wb = (wst + WIN16) // HOP16
        for i in range(nv):
            fe = db(frame_energy(out[i], nfr, HOP16))
            act = fe > max(fe.max() - 30, self.floor_db + 3)
            act = drop_short(fill_gaps(act, 25), 5)
            for a, b in runs(act):
                ov = np.minimum(wb, b) - np.maximum(wa, a)                  # frames of overlap
                m = (ov > 0) & valid[i]
                if not m.any():
                    continue
                w = (ov[m] / (wb[m] - wa[m])) * 10 ** (wdb[i, m] / 10)
                mean = (sc[i, m] * w[:, None]).sum(0) / w.sum()
                res[i, :, a:b] = self.ISLAND_MIX * mean[:, None] + (1 - self.ISLAND_MIX) * curves[i, :, a:b]
        return res

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
            if "memb" in c and out.shape[0] > 1:
                # one person alone: judge them from the original mix, which is cleaner
                msc = self._scores(c["memb"])                            # (nW, K)
                dom = c["wdb"].argmax(0)
                solo = c["solo"]
                sc[dom[solo], np.nonzero(solo)[0]] = msc[solo]
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
            curves = self._island_scores(out, sc, valid, c["wst"], c["wdb"], curves)
            labs = curves.argmax(1)                                          # (v, nfr)
            if self.DISTINCT and nv > 1 and K > 1:
                # voices the network pulled apart at the same moment are different people (only when
                # they really talk together and their voice prints differ - a single voice that the
                # network split in two must stay one person)
                from scipy.optimize import linear_sum_assignment
                fe = db(uniform_filter1d(np.stack([frame_energy(out[i], nfr, HOP16) for i in range(nv)]),
                                         10, axis=1, mode="nearest"))
                loudest = fe.max(0)
                act = (fe > max(fe.max() - 30, self.floor_db + 3)) & (fe > loudest[None] - self.SOLO_DB)
                E = c["emb"]
                cos = np.einsum("iwd,jwd->ijw", E, E)                       # (v, v, nW)
                cos_f = np.stack([np.stack([np.interp(fr, centers, cos[i, j]) for j in range(nv)])
                                  for i in range(nv)])                       # (v, v, nfr)
                cache = {}
                for t in np.nonzero(act.sum(0) >= 2)[0]:
                    idx = np.nonzero(act[:, t])[0]
                    sub = cos_f[np.ix_(idx, idx)][:, :, t]
                    if sub[np.triu_indices(len(idx), 1)].min() >= self.DIFFERENT_COS:
                        continue
                    if len(idx) > K:
                        idx = idx[np.argsort(-fe[idx, t])[:K]]
                    key = (tuple(idx), tuple(np.round(curves[idx, :, t].ravel(), 2)))
                    if key not in cache:
                        r, cidx = linear_sum_assignment(-curves[idx, :, t])
                        cache[key] = cidx
                    labs[idx, t] = cache[key]
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
        mw = getattr(self, "mix_win", None)
        if mw is not None and len(mw["st"]):
            # talking that wasn't separated (one clear voice): whole mix to that character
            u = np.clip(den, 0, 1)
            sc = self._scores(mw["emb"])                                    # (n, K)
            centers = (mw["st"] + WIN16 / 2) / HOP16
            fr = np.arange(self.nF) + 0.5
            curves = np.stack([np.interp(fr, centers, sc[:, j]) for j in range(K)])
            lab = majority(curves.argmax(0), K, 15) if K > 1 else np.zeros(self.nF, int)
            oh = np.zeros((K, self.nF), np.float32)
            oh[lab, np.arange(self.nF)] = 1
            oh = uniform_filter1d(oh, 3, axis=1, mode="nearest")
            A = np.repeat(oh, HOP16, axis=1)[:, :self.L16]
            est = est * u[None] + A * (self.mono16 * (1 - u))[None]
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
        if self.MASK_POWER != 1.0:
            P = P ** self.MASK_POWER
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
