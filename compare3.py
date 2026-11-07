"""Compare ready-made separators on a scene where three people talk at once (16 kHz mono)."""
import glob, numpy as np, soundfile as sf, torch, sys
from scipy.signal import resample_poly
mix, _ = sf.read('mix.flac', dtype='float32'); tr = [sf.read(f, dtype='float32')[0] for f in sorted(glob.glob('truth_*.flac'))]
act = np.stack([np.abs(t) > 3e-3 for t in tr]); three = act.sum(0) == 3
def sisdr(t, o):
    a = np.dot(o, t) / (np.dot(o, o) + 1e-12); return 10*np.log10((t**2).sum()/(((t-a*o)**2).sum()+1e-12))
def score(name, outs):
    r = [max(sisdr(t[three], o[three]) for o in outs) for t in tr]
    print('RESULT %-28s 3-way stretch per character: %s' % (name, [round(float(x), 1) for x in r]), flush=True)
def chunked(fn, sr_in, L):
    x = resample_poly(mix, sr_in, 16000).astype(np.float32) if sr_in != 16000 else mix
    outs = None; hop = L // 2; w = np.hanning(L)
    acc = None; wsum = np.zeros(len(x) + L)
    xp = np.pad(x, (0, L))
    for a in range(0, len(x), hop):
        seg = xp[a:a+L]; o = fn(seg)              # (n, L)
        if acc is None: acc = np.zeros((o.shape[0], len(x) + L))
        # (whole-scene permutation is fine here: the 3-way stretch fits in one piece)
        acc[:, a:a+L] += o * w; wsum[a:a+L] += w
    y = acc[:, :len(x)] / np.maximum(wsum[:len(x)], 1e-6)
    return [resample_poly(o, 16000, sr_in) [:len(mix)] if sr_in != 16000 else o for o in y]
from speechbrain.inference.separation import SepformerSeparation as S
for repo, sr in (('speechbrain/sepformer-wsj03mix', 8000), ('speechbrain/sepformer-libri3mix', 8000)):
    m = S.from_hparams(source=repo, savedir='pt/' + repo.split('/')[1], run_opts={'device': 'cpu'})
    st = int(np.argmax(three)); L = int(sr * 8)
    a = max(0, st - 32000); seg = mix[a:a + 128000]
    x = resample_poly(seg, sr, 16000).astype(np.float32)
    with torch.inference_mode(): o = m.separate_batch(torch.from_numpy(x[None]))[0].numpy().T
    o = [resample_poly(v, 16000, sr)[:len(seg)] for v in o]
    full = [np.zeros_like(mix) for _ in o]
    for f, v in zip(full, o): f[a:a+len(v)] = v
    score(repo, full)
