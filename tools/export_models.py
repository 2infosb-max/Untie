"""Build-machine step: download the SpeechBrain networks and freeze them into TorchScript files
inside untie/models, so the app needs only PyTorch (no SpeechBrain, no internet) to run them.

    python tools/export_models.py            # real pretrained networks (needs internet)
    python tools/export_models.py --random   # same architectures, random weights (offline test)
"""
import json
import os
import sys

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "untie", "models")
SEP_SECONDS = 4.0
ECAPA_BATCH = 16

SOURCES = {
    # MossFormer2 (ClearerVoice-Studio, Apache-2.0): ~20 dB on real two-voice speech where
    # SepFormer-WHAMR managed ~4 dB in our tests
    "sep2": ("alibabasglab/MossFormer2_SS_16K", 16000, 2),
    # three people at once: SepFormer trained on 3-speaker LibriMix beat every other ready-made
    # model on our three-way test (used only where three voices overlap)
    "sep3": ("speechbrain/sepformer-libri3mix", 8000, 3),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", 16000, None),
}


class SepWrap(torch.nn.Module):
    def __init__(self, encoder, masknet, decoder, n):
        super().__init__()
        self.encoder, self.masknet, self.decoder, self.n = encoder, masknet, decoder, n

    def forward(self, mix):                      # (1, T) -> (1, T, n)
        w = self.encoder(mix)
        m = self.masknet(w)
        outs = [self.decoder(w * m[i]) for i in range(self.n)]
        est = torch.stack(outs, -1)
        T = mix.shape[1]
        if est.shape[1] < T:
            est = torch.nn.functional.pad(est, (0, 0, 0, T - est.shape[1]))
        return est[:, :T, :]


class MossWrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, mix):                      # (1, T) -> (1, T, 2)
        outs = self.net(mix)
        return torch.stack([o[:, :mix.shape[1]] for o in outs], -1)


class EmbWrap(torch.nn.Module):
    def __init__(self, feats, norm, emb):
        super().__init__()
        self.feats, self.norm, self.emb = feats, norm, emb

    def forward(self, wav):                      # (B, T) -> (B, 192)
        lens = torch.ones(wav.shape[0])
        f = self.feats(wav)
        # the networks' own input normalisation is 'sentence' mean removal (std_norm off); written out
        # here because SpeechBrain's version uses .data, which freezing would turn into a constant
        f = f - f.mean(dim=1, keepdim=True)
        e = self.emb(f, lens).squeeze(1)
        return e / e.norm(dim=1, keepdim=True).clamp_min(1e-8)


def load_real(name):
    repo, sr, n = SOURCES[name]
    save = os.path.join(os.path.expanduser("~"), ".cache", "untie_sb", name)
    if name == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier
        m = EncoderClassifier.from_hparams(source=repo, savedir=save, run_opts={"device": "cpu"})
        mvn = m.mods.mean_var_norm
        assert mvn.norm_type == "sentence" and not mvn.std_norm, "unexpected normalisation"
        return EmbWrap(m.mods.compute_features, mvn, m.mods.embedding_model)
    if repo.startswith("alibabasglab/"):
        from clearvoice import ClearVoice
        cv = ClearVoice(task="speech_separation", model_names=[repo.split("/")[1]])
        return MossWrap(cv.models[0].model.cpu().eval())
    from speechbrain.inference.separation import SepformerSeparation
    m = SepformerSeparation.from_hparams(source=repo, savedir=save, run_opts={"device": "cpu"})
    return SepWrap(m.mods.encoder, m.mods.masknet, m.mods.decoder, n)


def load_random(name):
    """Same layers as the pretrained models, random weights - for testing the freezing offline."""
    torch.manual_seed(0)
    _, sr, n = SOURCES[name]
    if name == "ecapa":
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.processing.features import InputNormalization
        return EmbWrap(Fbank(n_mels=80), InputNormalization(norm_type="sentence", std_norm=False),
                       ECAPA_TDNN(80, lin_neurons=192, channels=[1024, 1024, 1024, 1024, 3072],
                                  kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                                  attention_channels=128))
    from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder, SBTransformerBlock
    enc = Encoder(kernel_size=16, out_channels=256)
    intra = SBTransformerBlock(num_layers=2, d_model=256, nhead=8, d_ffn=1024, dropout=0, use_positional_encoding=True,
                               norm_before=True)
    inter = SBTransformerBlock(num_layers=2, d_model=256, nhead=8, d_ffn=1024, dropout=0, use_positional_encoding=True,
                               norm_before=True)
    mask = Dual_Path_Model(num_spks=n, in_channels=256, out_channels=256, num_layers=2, K=250,
                           intra_model=intra, inter_model=inter, norm="ln", linear_layer_after_inter_intra=False,
                           skip_around_intra=True)
    dec = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
    return SepWrap(enc, mask, dec, n)


def check(a, b, what):
    err = (a - b).abs().max().item() / max(1e-6, b.abs().max().item())
    print(f"  {what}: frozen vs original relative difference {err:.2e}")
    if not err < 2e-3:
        raise SystemExit(f"FROZEN MODEL DOES NOT MATCH ({what})")


def main():
    random = "--random" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    mp = os.path.join(OUT, "meta.json")
    if only and os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    torch.manual_seed(1)
    for name, (repo, sr, n) in SOURCES.items():
        if only and name != only:
            continue
        print("freezing", name, "random" if random else repo)
        net = (load_random if random else load_real)(name).eval()
        with torch.inference_mode():
            if name == "ecapa":
                ex = torch.randn(ECAPA_BATCH, 16000) * 0.05
                traced = torch.jit.trace(net, ex, check_trace=False)
                ex2 = torch.randn(ECAPA_BATCH, 16000) * 0.1
                ex2[:, :4000] = 0
                check(traced(ex2), net(ex2), name)
                dim = int(net(ex).shape[1])
                meta[name] = {"sr": sr, "batch": ECAPA_BATCH, "length": 16000, "dim": dim, "source": repo}
            else:
                L = int((8.0 if name == "sep3" else SEP_SECONDS) * sr)
                ex = torch.randn(1, L) * 0.05
                traced = torch.jit.trace(net, ex, check_trace=False)
                t = torch.arange(L) / sr
                ex2 = (0.1 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.randn(L))[None]
                ex2[:, : L // 3] = 0
                check(traced(ex2), net(ex2), name)
                meta[name] = {"sr": sr, "length": L, "voices": n, "source": repo}
        traced = torch.jit.freeze(traced) if hasattr(torch.jit, "freeze") else traced
        with torch.inference_mode():
            check(traced(ex2), net(ex2), name + " (optimised)")
        torch.jit.save(traced, os.path.join(OUT, name + ".pt"))
    meta["random_weights"] = random
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("models written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
