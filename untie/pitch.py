"""Pitch (F0) tracking - YIN, vectorised with numpy. 10 ms frames."""
import numpy as np


def yin(x, sr, hop=None, fmin=55.0, fmax=600.0, thresh=0.15):
    """Returns (f0 Hz per frame, 0 where unvoiced; periodicity 0..1 per frame). Frame j is centred at
    j*hop + win/2 samples."""
    hop = hop or sr // 100
    win = int(sr / fmin * 2.2)
    tmax = min(int(sr / fmin), win - 1)
    tmin = max(2, int(sr / fmax))
    n = max(0, (len(x) - win) // hop + 1)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(1, keepdims=True)
    # difference function via FFT autocorrelation
    nf = 1 << int(np.ceil(np.log2(2 * win)))
    F = np.fft.rfft(fr, nf, axis=1)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(fr ** 2, 1)], 1)
    w = win - tmax
    e0 = cs[:, w] - cs[:, 0]
    taus = np.arange(tmax + 1)
    et = cs[:, taus + w] - cs[:, taus]
    r = np.fft.irfft(np.conj(np.fft.rfft(fr[:, :w], nf, axis=1)) * F, nf, axis=1)[:, :tmax + 1]
    d = e0[:, None] + et - 2 * r
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], 1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1)[None] / np.maximum(cm, 1e-12)
    f0 = np.zeros(n)
    per = np.zeros(n)
    sub = dn[:, tmin:]
    below = sub < thresh
    has = below.any(1)
    first = np.where(has, below.argmax(1), sub.argmin(1))
    # walk to the local minimum after the first dip
    for j in np.nonzero(has)[0]:
        t = first[j]
        while t + 1 < sub.shape[1] and sub[j, t + 1] < sub[j, t]:
            t += 1
        first[j] = t
    t = first + tmin
    val = dn[np.arange(n), t]
    # parabolic interpolation
    tl, tr = np.clip(t - 1, 1, tmax), np.clip(t + 1, 1, tmax)
    a, b, c = dn[np.arange(n), tl], val, dn[np.arange(n), tr]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / np.where(np.abs(den) > 1e-12, den, np.inf)
    tt = t + np.clip(off, -1, 1)
    energy = e0 / max(1, w)
    loud = 10 * np.log10(energy + 1e-12)
    voiced = (val < 0.35) & (loud > loud.max() - 45)
    f0[voiced] = sr / tt[voiced]
    per = np.clip(1 - val, 0, 1)
    return f0, per


def at_frames(f0, per, n_frames, offset_frames):
    """Re-index a YIN track (frame j centred at j*hop+win/2) to 10 ms frames starting at 0."""
    out_f = np.zeros(n_frames)
    out_p = np.zeros(n_frames)
    k = np.arange(len(f0)) + offset_frames
    m = (k >= 0) & (k < n_frames)
    out_f[k[m]] = f0[m]
    out_p[k[m]] = per[m]
    return out_f, out_p
