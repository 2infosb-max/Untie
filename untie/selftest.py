"""Proves a build really works: un-ties a scene whose true stems are known and scores the result.
Folder must hold mix.wav and truth_1.wav, truth_2.wav, ... (same length)."""
import glob
import os
import tempfile
import time

import numpy as np
import soundfile as sf

from . import engine
from .models import TorchModels


def sdr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12))


def run(folder, voices=2):
    mix, sr = sf.read(os.path.join(folder, "mix.wav"), dtype="float32", always_2d=True)
    truths = [sf.read(p, dtype="float32", always_2d=True)[0]
              for p in sorted(glob.glob(os.path.join(folder, "truth_*.wav")))]
    p = engine.Project(TorchModels, tempfile.mkdtemp())
    p.load(os.path.join(folder, "mix.wav"))
    p.settings["voices"] = voices
    t = time.time()
    p.analyse()
    t_an = time.time() - t
    p.find_characters()
    found = len(p.chars)
    if found != len(truths):
        print("SELFTEST auto-count found", found, "characters; using", len(truths))
        p.find_characters(len(truths))
    t = time.time()
    p.untie()
    t_un = time.time() - t
    stems = [sf.read(q, dtype="float32", always_2d=True)[0] for q in p.stem_paths()]
    best, base = [], []
    used = set()
    for tr in truths:
        sc = [(sdr(tr, s), i) for i, s in enumerate(stems[:-1]) if i not in used]
        v, i = max(sc)
        used.add(i)
        best.append(round(float(v), 1))
        base.append(round(float(sdr(tr, mix)), 1))
    gain = [round(b - a, 1) for a, b in zip(base, best)]
    print("SELFTEST seconds=%.1f analyse=%.1fs untie=%.1fs" % (len(mix) / sr, t_an, t_un))
    print("SELFTEST characters_found=%d expected=%d" % (found, len(truths)))
    print("SELFTEST stem_SDR=%s mixture_SDR=%s improvement=%s" % (best, base, gain))
    print("SELFTEST null_db=%.1f" % p.null_db)
    outs = p.export(tempfile.mkdtemp(), aaf=True)
    assert any(o.endswith(".aaf") for o in outs)
    assert any(o.endswith("_voice_analysis.md") for o in outs), "voice analysis missing"
    print("SELFTEST exported", [os.path.basename(o) for o in outs])
    return {"found": found, "expected": len(truths), "sdr": best, "gain": gain, "null_db": p.null_db}
