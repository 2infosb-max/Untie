"""End-to-end test of the Un-tie engine with a stand-in separator (no neural networks needed).
Checks: characters are found, each stem matches its true voice, stems null against the
original, and timeline fixes re-render correctly."""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synth import scene, sdr  # noqa: E402
from untie import audio_io, engine  # noqa: E402
from untie.engine import resample  # noqa: E402
from untie.models import OracleModels  # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    sr = 48000
    mix, truth, noise = scene(sr)
    path = os.path.join(tmp, "scene.wav")
    sf.write(path, mix, sr, subtype="FLOAT")
    src16 = np.stack([resample(t.mean(1), sr, 16000) for t in truth])
    p = engine.Project(lambda: OracleModels(src16), os.path.join(tmp, "work"))
    info = p.load(path)
    print("loaded", info)
    p.analyse(lambda pct, msg, eta: None)
    print("chunks", len(p.chunks))
    chars = p.find_characters(3)   # auto-count is tuned for the real voice network; tested on the Mac build
    print("characters found:", len(chars))
    assert len(chars) == 3, len(chars)
    p.untie()
    print("null test dB (float):", round(p.null_db, 1), " 24-bit:", round(p.null_db_24, 1))
    assert p.null_db < -100, p.null_db
    stems = [sf.read(q, dtype="float32", always_2d=True)[0] for q in p.stem_paths()]
    total = sum(s.astype(np.float64) for s in stems)
    print("sum of 24-bit stems vs original:", 20 * np.log10(np.abs(total - mix).max()))
    assert np.abs(total - mix).max() < 1e-5
    res = []
    for t in truth:
        scores = [sdr(t, s) for s in stems[:-1]]
        res.append(max(scores))
    base = [sdr(t, mix) for t in truth]
    print("stem quality (SDR dB) per character:", [round(r, 1) for r in res],
          " (mixture alone:", [round(b, 1) for b in base], ")")
    assert min(res) > 12, res
    # where only one person talks, their stem must be the original, untouched
    a, b = int(0.5 * sr), int(2.0 * sr)
    j = int(np.argmax([np.abs(s[a:b]).sum() for s in stems[:-1]]))
    err = np.abs(stems[j][a:b] - mix[a:b]).max()
    print("solo line passthrough error:", err)
    assert err < 1e-5, err
    # a fix on the timeline: give 9.8-11.3 s to a different character, then undo it
    wrong = p.chars[(j + 1) % 3]["id"]
    before = sf.read(p.stem_paths()[0], dtype="float32")[0].copy()
    p.add_edit("assign", 9.8, 11.3, [wrong])
    stems2 = [sf.read(q, dtype="float32", always_2d=True)[0] for q in p.stem_paths()]
    k = [c["id"] for c in p.chars].index(wrong)
    a, b = int(10.2 * sr), int(11.0 * sr)
    assert np.abs(stems2[k][a:b] - mix[a:b]).max() < 1e-5
    assert np.abs(sum(s.astype(np.float64) for s in stems2) - mix).max() < 1e-5
    p.remove_edit(0)
    after = sf.read(p.stem_paths()[0], dtype="float32")[0]
    assert np.abs(after - before).max() < 1e-6, np.abs(after - before).max()
    # BWF timecode survives
    tc = 3600 * 10 * sr
    path2 = os.path.join(tmp, "tc.wav")
    w = audio_io.WavWriter(path2, sr, 2, len(mix), time_ref=tc)
    w.write_at(0, mix)
    w.close()
    assert audio_io.read_bext_time_ref(path2) == tc
    p2 = engine.Project(lambda: OracleModels(src16), os.path.join(tmp, "work"))
    p2.load(path2)
    p2.analyse()
    p2.find_characters(3)
    p2.untie()
    outs = p2.export(tmp)
    assert all(audio_io.read_bext_time_ref(o) == tc for o in outs)
    outs = p2.export(tmp, fmt="float32")
    x = sf.read(outs[0], dtype="float32")[0]
    assert len(x) == len(mix)
    print("export ok:", [os.path.basename(o) for o in outs])
    # mono, 44.1 kHz, and a selection-made character
    m = mix.mean(1)
    sf.write(os.path.join(tmp, "mono.wav"), resample(m, sr, 44100), 44100, subtype="PCM_24")
    p3 = engine.Project(lambda: OracleModels(src16), os.path.join(tmp, "work"))
    p3.load(os.path.join(tmp, "mono.wav"))
    p3.analyse()
    p3.add_character("Hero", 0.3, 2.4)
    p3.add_character("Villain", 2.9, 4.1)
    p3.add_character("Friend", 6.5, 7.5)
    p3.untie()
    assert p3.null_db < -100
    print("mono 44.1k null:", round(p3.null_db, 1))
    print("ALL ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
