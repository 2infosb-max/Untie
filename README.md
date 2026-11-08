# Un-tie by Sound Brewery

Mac app that recovers per-character dialogue stems from a mixed mono/stereo dialogue track, for when the original ADR/dubbing files are lost.

## How it works

1. **Voice activity** finds every stretch of talking, and a voice print of the mix is taken every 0.25 s. Only stretches that aren't clearly one known voice (overlaps, unclear) go through the slow separation; the option *Separate every line* runs it everywhere.
2. **Separation**: a MossFormer2 network (ClearerVoice-Studio, 16 kHz) splits each 4 s piece into 2 voices; the voices are scaled and polarity-matched so together they rebuild the mix.
3. **Who is who**: an ECAPA-TDNN speaker-identity network fingerprints the voices every 0.25 s - from the original mix where one person talks alone (cleaner), from the separated voices in overlaps. Characters are found by clustering ~2 s solo stretches. Two separated voices talking at once with different fingerprints always go to two different characters.
4. **Decision every 10 ms**: silence, one character alone, or an overlap.
5. **Stems from the original**: each 10 ms frame of the original full-resolution spectrum is shared out:
   - one character alone: 100% to them (a plain fade, so the sound is untouched),
   - overlap: shared by each separated voice's energy per frequency; above the network's 8 kHz range the share at the top of its range is carried up, so the original air and top end stay,
   - silence: Residual.
   The shares add up to exactly 1, so the stems sum back to the original (null test about −140 dB).

Three people at once: where the two-voice split still holds two people, a SepFormer trained on 3-speaker LibriMix splits that stretch. On a real-speech three-way scene this raised the share of each stem that belongs to the right character in overlaps from 84/76/69% to 96/79/88%.

**Splitting intensity** (0-100%): in overlaps, each character's harmonics (from their pitch) pull those frequencies to them, and a strictness gate sends uncertain frequencies to the Residual rather than into another character. At 100% on the three-way scene: 100/99/98% of each stem belongs to the right character across all talking (99/92/92% in overlaps), at the cost of thinner voices in overlaps.

**Voice analysis** (`untie/voice_profile.py`): per character, from solo stretches - F0 (YIN), formants (LPC), vocal-tract length, H1-H2, spectral tilt, HNR, jitter/shimmer, nasal index, fry, vibrato, dynamics, consonant energy, speech rate - classified into the four sections plus an executive summary with timecodes and a separation matrix.

Timeline fixes (only X / overlap of / nobody) re-render only the affected stretch.

Export: WAV stems (24-bit or 32-bit float, BWF timecode kept) and/or a Pro Tools AAF (pyaaf2): one composition, one mono track per stem channel, embedded 24-bit audio, timecode start from the source stamp with the sub-frame remainder kept as filler so sync is sample-accurate; optional original-mix guide tracks.

## Build

GitHub Actions (`.github/workflows/build-dmg.yml`) builds an Apple-silicon and an Intel `.dmg` on every push to `main`, after:
- `tests/engine_test.py`: the full pipeline with a stand-in separator (null test, untouched solo lines, fixes, timecode),
- `tools/export_models.py`: downloads the networks and freezes them to TorchScript (checked against the originals),
- a real-speech scene made with macOS voices (4 characters with overlaps), un-tied and scored before and after packaging.

Networks: alibabasglab/MossFormer2_SS_16K (ClearerVoice-Studio) and speechbrain/spkrec-ecapa-voxceleb, both Apache-2.0.
