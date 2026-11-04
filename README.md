# Un-tie by Sound Brewery

Mac app that recovers per-character dialogue stems from a mixed mono/stereo dialogue track, for when the original ADR/dubbing files are lost.

## How it works

1. **Voice activity** finds every stretch of talking.
2. **Separation**: a SepFormer network (SpeechBrain, trained on noisy reverberant speech, 16 kHz) splits each 8 s piece into 2 (or 3) voices.
3. **Who is who**: an ECAPA-TDNN speaker-identity network fingerprints each separated voice every 0.25 s and matches it to a character. Two voices separated at the same moment are always given to two different characters.
4. **Decision every 10 ms**: silence, one character alone, or an overlap.
5. **Stems from the original**: each 10 ms frame of the original full-resolution spectrum is shared out:
   - one character alone: 100% to them (a plain fade, so the sound is untouched),
   - overlap: shared by each separated voice's energy per frequency; above the network's 8 kHz range the share at the top of its range is carried up, so the original air and top end stay,
   - silence: Residual.
   The shares add up to exactly 1, so the stems sum back to the original (null test about −140 dB).

Timeline fixes (only X / overlap of / nobody) re-render only the affected stretch.

Export: WAV stems (24-bit or 32-bit float, BWF timecode kept) and/or a Pro Tools AAF (pyaaf2): one composition, one mono track per stem channel, embedded 24-bit audio, timecode start from the source stamp with the sub-frame remainder kept as filler so sync is sample-accurate; optional original-mix guide tracks.

## Build

GitHub Actions (`.github/workflows/build-dmg.yml`) builds an Apple-silicon and an Intel `.dmg` on every push to `main`, after:
- `tests/engine_test.py`: the full pipeline with a stand-in separator (null test, untouched solo lines, fixes, timecode),
- `tools/export_models.py`: downloads the networks and freezes them to TorchScript (checked against the originals),
- a real-speech scene made with macOS voices (4 characters with overlaps), un-tied and scored before and after packaging.

Networks: speechbrain/sepformer-whamr16k, speechbrain/sepformer-libri3mix, speechbrain/spkrec-ecapa-voxceleb (Apache-2.0).
