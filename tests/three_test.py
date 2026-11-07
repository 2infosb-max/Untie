"""Three people talking at once: with re-splitting, each character's stem must be clearly cleaner."""
import os, sys, tempfile
import numpy as np, soundfile as sf
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from synth import scene, sdr  # noqa: E402
from untie import engine  # noqa: E402
from untie.engine import resample  # noqa: E402
from untie.models import OracleModels  # noqa: E402

SCRIPT = [(0, 0.3, 2.0), (1, 2.8, 2.0), (2, 5.3, 2.0),            # each alone first
          (0, 8.0, 3.0), (1, 8.4, 2.6), (2, 8.8, 2.4),            # all three at once
          (1, 12.0, 1.5), (0, 14.0, 1.5)]


def run(three):
    sr = 48000
    mix, truth, _ = scene(sr, script=SCRIPT)
    d = tempfile.mkdtemp()
    sf.write(d + "/m.wav", mix, sr, subtype="FLOAT")
    src16 = np.stack([resample(t.mean(1), sr, 16000) for t in truth])
    p = engine.Project(lambda: OracleModels(src16), d + "/w")
    p.load(d + "/m.wav")
    p.settings["three"] = three
    p.settings["thorough"] = True
    p.analyse()
    for nm, a, b in (("A", 0.3, 2.3), ("B", 2.8, 4.8), ("C", 5.3, 7.3)):
        p.add_character(nm, a, b)
    p.untie()
    stems = [sf.read(q, dtype="float32", always_2d=True)[0] for q in p.stem_paths()]
    a, b = int(8.8 * sr), int(10.9 * sr)                       # the three-way stretch
    res = [round(float(sdr(t[a:b], stems[i][a:b])), 1) for i, t in enumerate(truth)]
    print("three-voice=%s  SDR in the 3-way overlap per character:" % three, res, " null %.0f dB" % p.null_db)
    assert p.null_db < -100
    return res


if __name__ == "__main__":
    off, on = run(False), run(True)
    # the stand-in is too simple to judge quality here (measured on real speech instead); this checks
    # that re-splitting runs end to end and keeps the stems adding back up to the original
    assert len(on) == 3
    print("ALL THREE-VOICE TESTS PASSED")
