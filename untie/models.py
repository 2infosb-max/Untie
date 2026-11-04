"""The two neural networks Un-tie uses, loaded from the app bundle.

sep2 / sep3 : SepFormer (SpeechBrain, Apache-2.0) traced to TorchScript - splits a mixture
              into 2 or 3 voices.
ecapa       : ECAPA-TDNN speaker embedding (SpeechBrain, Apache-2.0) - "whose voice is this?"
"""
import json
import os
import sys
import threading

import numpy as np

SR16 = 16000
WIN16 = 16000          # 1 s windows for voice identity


def models_dir():
    base = getattr(sys, "_MEIPASS", None)
    cands = []
    if base:
        cands.append(os.path.join(base, "untie", "models"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    if os.environ.get("UNTIE_MODELS"):
        cands.insert(0, os.environ["UNTIE_MODELS"])
    for c in cands:
        if os.path.exists(os.path.join(c, "meta.json")):
            return c
    raise FileNotFoundError("The Un-tie voice models are missing from this copy of the app.")


def _resample(x, a, b):
    if a == b:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(a, b)
    return resample_poly(x, b // g, a // g, axis=-1).astype(np.float32)


class TorchModels:
    def __init__(self, folder=None):
        self.folder = folder or models_dir()
        with open(os.path.join(self.folder, "meta.json")) as f:
            self.meta = json.load(f)
        self._m = {}
        self._lock = threading.Lock()
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
        self.torch = torch

    def _get(self, name):
        with self._lock:
            if name not in self._m:
                m = self.torch.jit.load(os.path.join(self.folder, name + ".pt"), map_location="cpu")
                m.eval()
                self._m[name] = m
            return self._m[name]

    def chunk_len16(self, voices):
        m = self.meta["sep%d" % voices]
        return int(round(m["length"] * SR16 / m["sr"]))

    def model_sr(self, voices):
        return self.meta["sep%d" % voices]["sr"]

    def separate(self, x16, voices=2):
        """x16: (L16,) float32 at 16 kHz, L16 == chunk_len16(voices). Returns (voices, L16)."""
        m = self.meta["sep%d" % voices]
        x = _resample(x16, SR16, m["sr"])
        n = m["length"]
        x = np.pad(x, (0, max(0, n - len(x))))[:n]
        net = self._get("sep%d" % voices)
        with self.torch.inference_mode():
            y = net(self.torch.from_numpy(np.ascontiguousarray(x[None], dtype=np.float32))).numpy()[0].T
        y = _resample(y, m["sr"], SR16)
        out = np.zeros((y.shape[0], len(x16)), np.float32)
        k = min(len(x16), y.shape[1])
        out[:, :k] = y[:, :k]
        return out

    def embed(self, wins):
        """wins: (B, 16000) float32. Returns (B, D) unit vectors."""
        m = self.meta["ecapa"]
        bs = m["batch"]
        net = self._get("ecapa")
        res = []
        for i in range(0, len(wins), bs):
            b = wins[i:i + bs]
            n = len(b)
            if n < bs:
                b = np.concatenate([b, np.zeros((bs - n, b.shape[1]), np.float32)])
            # each window at a common loudness, so quiet lines are recognised as well as loud ones
            r = np.sqrt((b ** 2).mean(1, keepdims=True)) + 1e-6
            b = (b * (0.05 / r)).astype(np.float32)
            with self.torch.inference_mode():
                e = net(self.torch.from_numpy(b)).numpy()[:n]
            res.append(e)
        e = np.concatenate(res) if res else np.zeros((0, m["dim"]), np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


class OracleModels:
    """Stand-in used by the tests: 'separates' using the true sources and recognises voices by
    their spectrum. Lets the whole pipeline be tested without the neural networks."""

    def __init__(self, sources16, chunk_s=8.0, leak=0.0):
        self.src = sources16          # (S, N16) true voices at 16 kHz
        self.L = int(chunk_s * SR16)
        self.leak = leak
        self.meta = {"sep2": {"sr": SR16}, "sep3": {"sr": SR16}}
        self.pos = None

    def chunk_len16(self, voices):
        return self.L

    def model_sr(self, voices):
        return SR16

    def locate(self, start):
        self.pos = start

    def separate(self, x16, voices=2):
        s = self.pos
        seg = np.zeros((self.src.shape[0], len(x16)), np.float32)
        k = max(0, min(len(x16), self.src.shape[1] - s))
        seg[:, :k] = self.src[:, s:s + k]
        tot = seg.sum(0)
        gain = float(np.dot(x16, tot) / (np.dot(tot, tot) + 1e-12))     # the caller scales its input
        # like a real separator: at each moment the loudest voices go to the outputs,
        # and an output can carry different people at different times
        blk = 1600
        nb = int(np.ceil(len(x16) / blk))
        pad = np.zeros((seg.shape[0], nb * blk), np.float32)
        pad[:, :len(x16)] = seg
        e = (pad.reshape(seg.shape[0], nb, blk) ** 2).sum(2)                # (S, nb)
        # a real separator keeps each voice in the same output while it talks
        S = seg.shape[0]
        slot_of = {}
        sel = np.zeros((voices, S, nb), np.float32)
        thr = e.max() * 1e-5
        for t in range(nb):
            act = [j for j in np.argsort(-e[:, t]) if e[j, t] > thr][:voices]
            slot_of = {j: q for j, q in slot_of.items() if j in act}
            free = [q for q in range(voices) if q not in slot_of.values()]
            for j in act:
                if j not in slot_of:
                    slot_of[j] = free.pop(0)
            for j, q in slot_of.items():
                sel[q, j, t] = 1
        out = np.zeros((voices, len(x16)), np.float32)
        for q in range(voices):
            w = np.repeat(sel[q], blk, axis=1)[:, :len(x16)]
            out[q] = (w * seg).sum(0)
        if self.leak:
            out = out * (1 - self.leak) + self.leak * out[::-1]
        return out * gain

    def embed(self, wins):
        # pitch + formant fingerprint: fine log spectrum 60-3000 Hz, smoothed
        spec = np.abs(np.fft.rfft(wins * np.hanning(wins.shape[1]), axis=1))[:, 60:3000]
        f = np.log(spec + 1e-4)
        from scipy.ndimage import uniform_filter1d
        f = uniform_filter1d(f, 9, axis=1)[:, ::4]
        f = f - f.mean(1, keepdims=True)
        return (f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
