"""Runs the app page with the stand-in separator, for testing the interface in a browser."""
import os, sys, tempfile
import numpy as np, soundfile as sf
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from synth import scene
from untie import app as A
from untie.engine import resample
from untie.models import OracleModels
sr = 48000
mix, truth, _ = scene(sr)
d = tempfile.mkdtemp(); path = os.path.join(d, "Scene 12 Take 3.wav")
from untie import audio_io
w = audio_io.WavWriter(path, sr, 2, len(mix), time_ref=int(sr * (10 * 3600 + 12 * 60 + 4.5))); w.write_at(0, mix); w.close()
src16 = np.stack([resample(t.mean(1), sr, 16000) for t in truth])
A.MODELS_FACTORY[0] = lambda: OracleModels(src16)
print("FILE", path, flush=True)
A.app.run(host="127.0.0.1", port=int(sys.argv[1]), threaded=True)
