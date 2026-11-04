"""Synthetic 'characters' for testing: harmonic voices with their own pitch and formants."""
import numpy as np

VOICES = [  # f0, formants
    (105, (600, 1100, 2600)),
    (210, (850, 1900, 3100)),
    (155, (400, 2200, 2900)),
    (290, (700, 1500, 3500)),
]


def voice(sr, seconds, f0, formants, seed):
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # syllables: on/off envelope
    env = np.zeros(n)
    pos = 0.0
    while pos < seconds:
        d = rng.uniform(0.12, 0.35)
        a, b = int(pos * sr), int(min(seconds, pos + d) * sr)
        env[a:b] = np.hanning(b - a) * rng.uniform(0.5, 1.0)
        pos += d + rng.uniform(0.0, 0.12)
    vib = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(3, 6) * t) + 0.02 * np.cumsum(rng.normal(0, 1, n)) / np.sqrt(n * 50)
    ph = 2 * np.pi * np.cumsum(f0 * vib) / sr
    y = np.zeros(n)
    h = 1
    while h * f0 < min(sr / 2 - 500, 15000):
        fh = h * f0
        g = sum(np.exp(-((fh - F) / 180.0) ** 2) for F in formants) + 0.02 / h
        y += g * np.sin(h * ph)
        h += 1
    y += 0.02 * rng.normal(0, 1, n) * (env > 0.05)         # breath / air, full band
    y = y * env
    return (y / (np.abs(y).max() + 1e-9) * 0.3).astype(np.float32)


def scene(sr=48000, script=None, chars=3, pans=(-0.6, 0.6, 0.0, -0.2), noise_db=-62, seed=1):
    """Return (mix (N,2), truth list of (N,2)). Script: list of (char, start, dur)."""
    if script is None:
        script = [(0, 0.3, 2.2), (1, 2.9, 2.0), (0, 4.2, 1.8),      # 4.2-4.9 overlap 0+1
                  (2, 6.5, 2.4), (1, 7.6, 1.6),                     # overlap 2+1
                  (0, 9.8, 1.5), (2, 11.5, 1.8), (1, 12.0, 1.0), (0, 13.6, 1.2)]
    total = max(s + d for _, s, d in script) + 0.6
    N = int(total * sr)
    truth = [np.zeros((N, 2), np.float32) for _ in range(chars)]
    for k, (c, s, d) in enumerate(script):
        f0, F = VOICES[c]
        v = voice(sr, d, f0 * (1 + 0.04 * ((k % 3) - 1)), F, seed * 100 + k)
        a = int(s * sr)
        p = pans[c]
        truth[c][a:a + len(v), 0] += v * np.sqrt((1 - p) / 2)
        truth[c][a:a + len(v), 1] += v * np.sqrt((1 + p) / 2)
    rng = np.random.default_rng(seed)
    noise = (rng.normal(0, 1, (N, 2)) * 10 ** (noise_db / 20)).astype(np.float32)
    mix = sum(truth) + noise
    return mix.astype(np.float32), truth, noise


def sdr(ref, est):
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-12))
