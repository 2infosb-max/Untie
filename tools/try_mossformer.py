"""Build-machine experiment: compare MossFormer2 (ClearerVoice, Apache-2.0) with SepFormer-WHAMR
on real speech, and freeze MossFormer2 to TorchScript. Usage: try_mossformer.py SCENE_DIR OUT_DIR"""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def sisdr(t, o):
    a = np.dot(o, t) / (np.dot(o, o) + 1e-12)
    e = a * o
    return 10 * np.log10((t ** 2).sum() / (((t - e) ** 2).sum() + 1e-12))


class Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        outs = self.net(x)
        return torch.stack([o[:, :x.shape[1]] for o in outs], -1)


def main(scene, out):
    os.makedirs(out, exist_ok=True)
    from clearvoice import ClearVoice
    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    net = Wrap(cv.models[0].model.cpu().eval())
    truths = [sf.read(f, dtype="float32", always_2d=True)[0].mean(1)
              for f in sorted(glob.glob(os.path.join(scene, "truth_*.wav")))]
    t16 = [resample_poly(t, 1, 3).astype(np.float32) for t in truths]
    seps = {"mossformer2": lambda x: net(torch.from_numpy(x[None])).detach().numpy()[0].T}
    try:
        sys.path.insert(0, os.getcwd())
        from untie.models import TorchModels
        M = TorchModels()
        seps["sepformer"] = lambda x: M.separate(np.pad(x, (0, max(0, 128000 - len(x))))[:128000], 2)[:, :len(x)]
    except Exception as e:  # noqa: BLE001
        print("sepformer not available:", e)
    a = t16[0][:64000]
    b = t16[3][int(14.8 * 16000):int(14.8 * 16000) + 64000]
    for name, fn in seps.items():
        for gb in (1.0, 0.4):
            m = (a + gb * b).astype(np.float32)
            m = m * (0.05 / np.sqrt((m ** 2).mean()))
            t0 = time.time()
            with torch.inference_mode():
                o = fn(m)
            dt = time.time() - t0
            r = [max(sisdr(t, x) for x in o) for t in (a, gb * b)]
            print("TRY %-12s gain %.1f  SI-SDR %.1f / %.1f dB   %.2fs for 4 s" % (name, gb, r[0], r[1], dt))
    for L in (32000, 64000):
        ex = torch.randn(1, L) * 0.05
        with torch.inference_mode():
            tr = torch.jit.trace(net, ex, check_trace=False)
            x2 = torch.from_numpy(np.pad((a + b)[:L] * 3, (0, 0))[None].astype(np.float32))
            d = (tr(x2) - net(x2)).abs().max().item() / net(x2).abs().max().item()
        print("TRY traced %d: relative difference %.2e" % (L, d))
        torch.jit.save(tr, os.path.join(out, "moss_%d.pt" % L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
