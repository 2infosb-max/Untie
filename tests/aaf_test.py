"""Checks the Pro Tools AAF: one track per stem channel, embedded audio identical to the stems,
starting at the right timecode."""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synth import scene  # noqa: E402
from untie import aaf_export, audio_io, engine  # noqa: E402
from untie.engine import resample  # noqa: E402
from untie.models import OracleModels  # noqa: E402


def main():
    import aaf2
    tmp = tempfile.mkdtemp()
    sr = 48000
    mix, truth, _ = scene(sr)
    path = os.path.join(tmp, "Reel 2 Sc 14.wav")
    tref = sr * (10 * 3600 + 2 * 60 + 3) + sr // 2          # 10:02:03 + half a second
    w = audio_io.WavWriter(path, sr, 2, len(mix), time_ref=tref)
    w.write_at(0, mix)
    w.close()
    src16 = np.stack([resample(t.mean(1), sr, 16000) for t in truth])
    p = engine.Project(lambda: OracleModels(src16), os.path.join(tmp, "work"))
    p.load(path)
    p.analyse()
    for nm, a, b in (("Hero", 0.3, 2.4), ("Character 2", 2.9, 4.1), ("Character 3", 6.5, 7.5)):
        p.add_character(nm, a, b)
    p.untie()

    assert aaf_export.default_start_tc(tref, sr, 25) == ("10:02:03:12", 960), aaf_export.default_start_tc(tref, sr, 25)
    assert aaf_export.default_start_tc(None, sr, 25) == ("01:00:00:00", 0)
    assert aaf_export.default_start_tc(sr * 3600, sr, 23.976)[0] == "00:59:56:09"

    out = os.path.join(tmp, "out")
    os.makedirs(out)
    files = p.export(out, aaf=True, wavs=False, fps=25.0)
    aaf_path = [f for f in files if f.endswith(".aaf")][0]
    print("AAF:", os.path.basename(aaf_path), round(os.path.getsize(aaf_path) / 1e6, 1), "MB")

    with aaf2.open(aaf_path, "r") as f:
        top = list(f.content.toplevel())
        assert len(top) == 1
        comp = top[0]
        tc = [s for s in comp.slots if s.segment.media_kind == "Timecode"]
        snd = [s for s in comp.slots if s.segment.media_kind == "Sound"]
        assert tc and tc[0].segment.start == aaf_export.tc_to_frames("10:02:03:12", 25), tc[0].segment.start
        names = [s.name for s in snd]
        print("tracks:", names)
        assert names == ["Hero L", "Hero R", "Character 2 L", "Character 2 R", "Character 3 L", "Character 3 R",
                         "Residual L", "Residual R", "Original mix L", "Original mix R"], names
        stems = [sf.read(q, dtype="float32", always_2d=True)[0] for q in p.stem_paths()] + [mix]
        for k, s in enumerate(snd):
            fill, clip = s.segment.components                       # 960-sample lead, then the stem
            assert fill.length == 960, fill.length
            assert clip.length == len(mix), (clip.length, len(mix))
            master = clip.mob
            src = master.slots[0].segment.mob
            wav = os.path.join(tmp, "x%d.wav" % k)
            src.export_audio(wav)
            x = sf.read(wav, dtype="float32")[0]
            ref = audio_io.quantize24(stems[k // 2][:, k % 2])
            err = np.abs(x - ref).max()
            assert err < 1e-6, (names[k], err)
    print("embedded audio matches every stem, timecode correct")

    # 23.976 and a typed start timecode, WAVs + AAF together, no residual, no guide
    out2 = os.path.join(tmp, "out2")
    os.makedirs(out2)
    files = p.export(out2, aaf=True, wavs=True, fps=23.976, start_tc="01:00:00:00", residual=False, guide=False)
    assert len([f for f in files if f.endswith(".wav")]) == 3
    with aaf2.open([f for f in files if f.endswith(".aaf")][0], "r") as f:
        comp = list(f.content.toplevel())[0]
        snd = [s for s in comp.slots if s.segment.media_kind == "Sound"]
        tcs = [s for s in comp.slots if s.segment.media_kind == "Timecode"][0]
        assert len(snd) == 6 and tcs.segment.start == 24 * 3600 and tcs.segment.fps == 24
        assert str(tcs.edit_rate) == "24000/1001", tcs.edit_rate
    try:
        aaf_export.tc_to_frames("25:00:00:00", 25)
        raise AssertionError("bad timecode accepted")
    except ValueError:
        pass
    print("ALL AAF TESTS PASSED")


if __name__ == "__main__":
    main()
