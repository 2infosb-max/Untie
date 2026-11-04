"""Mac build machine only: makes a test scene with real speech from macOS's built-in voices.
Four characters, with two-voice overlaps, mixed to 48 kHz stereo. Writes mix.wav + truth_N.wav."""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PREFER = ["Daniel", "Samantha", "Rishi", "Karen", "Moira", "Fred", "Tessa", "Alex", "Ralph", "Kathy", "Albert"]
LINES = [
    (0, "Where were you last night? I called you four times and nobody picked up."),
    (1, "I told you already, I was at the studio until late. The mix ran over."),
    (0, "The mix ran over. Every single time, the mix runs over."),
    (2, "Can you two please stop shouting, the neighbours can hear everything."),
    (1, "Stay out of this, it has nothing to do with you."),
    (3, "Dinner is ready, come and eat before it gets cold."),
    (2, "See, even she is tired of this argument."),
    (0, "Fine. We will talk about it after dinner, and you will tell me the truth."),
    (3, "Nobody is talking about anything until the plates are empty."),
]
# start times (s): several lines start before the previous one ends -> overlaps
STARTS = [0.3, 4.0, 7.2, 9.6, 12.4, 15.0, 16.6, 19.4, 23.2]


def voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = [re.split(r"\s{2,}", l.strip())[0] for l in out.splitlines() if " en_" in l]
    pick = [v for v in PREFER if v in names]
    pick += [v for v in names if v not in pick]
    if len(pick) < 4:
        raise SystemExit("not enough voices: %s" % names)
    return pick[:4]


def say(voice, text, tmp):
    f = os.path.join(tmp, "x.aiff")
    subprocess.run(["say", "-v", voice, "-o", f, "--data-format=BEI16@22050", text], check=True)
    x, sr = sf.read(f, dtype="float32")
    return resample_poly(x, 320, 147).astype(np.float32)          # 22050 -> 48000


def main(folder):
    os.makedirs(folder, exist_ok=True)
    vs = voices()
    print("voices:", vs)
    tmp = tempfile.mkdtemp()
    sr = 48000
    clips = [(c, STARTS[i], say(vs[c], t, tmp)) for i, (c, t) in enumerate(LINES)]
    N = int(max(s * sr + len(x) for _, s, x in clips)) + sr
    pans = [-0.5, 0.5, 0.15, -0.15]
    truth = [np.zeros((N, 2), np.float32) for _ in range(4)]
    for c, s, x in clips:
        a = int(s * sr)
        g = 0.5 / (np.abs(x).max() + 1e-9)
        truth[c][a:a + len(x), 0] += x * g * np.sqrt((1 - pans[c]) / 2)
        truth[c][a:a + len(x), 1] += x * g * np.sqrt((1 + pans[c]) / 2)
    rng = np.random.default_rng(0)
    mix = sum(truth) + rng.normal(0, 10 ** (-60 / 20), (N, 2)).astype(np.float32)
    sf.write(os.path.join(folder, "mix.wav"), mix, sr, subtype="PCM_24")
    for i, t in enumerate(truth):
        sf.write(os.path.join(folder, "truth_%d.wav" % (i + 1)), t, sr, subtype="FLOAT")
    ov = 0.0
    act = np.stack([np.abs(t).max(1) > 1e-3 for t in truth])
    ov = (act.sum(0) >= 2).mean() * N / sr
    print("scene: %.1f s, about %.1f s of overlapping speech" % (N / sr, ov))


if __name__ == "__main__":
    main(sys.argv[1])
