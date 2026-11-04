"""Pro Tools AAF: one sequence with every stem on its own track, audio embedded, starting at the
chosen timecode. Stereo stems become two mono tracks (L, R) - the way Pro Tools imports AAF."""
import os
import shutil
import tempfile

import soundfile as sf

RATES = {23.976: ("24000/1001", 24), 24.0: ("24", 24), 25.0: ("25", 25),
         29.97: ("30000/1001", 30), 30.0: ("30", 30)}


def tc_to_frames(tc, fps_nominal):
    parts = [int(p) for p in str(tc).replace(";", ":").replace(".", ":").split(":")]
    if len(parts) != 4 or not (0 <= parts[0] < 24 and 0 <= parts[1] < 60 and 0 <= parts[2] < 60
                               and 0 <= parts[3] < fps_nominal):
        raise ValueError("Start timecode must look like 01:00:00:00")
    h, m, s, f = parts
    return ((h * 60 + m) * 60 + s) * fps_nominal + f


def frames_to_tc(frames, fps_nominal):
    f = frames % fps_nominal
    s = frames // fps_nominal
    return "%02d:%02d:%02d:%02d" % ((s // 3600) % 24, (s // 60) % 60, s % 60, f)


def default_start_tc(time_ref, sr, fps):
    """The source's BWF stamp as timecode (01:00:00:00 if it has none), and how many samples
    after that frame the audio really starts (the stamp is rarely exactly on a frame)."""
    rate, nominal = RATES.get(round(float(fps), 3), RATES[25.0])
    if not time_ref:
        return "01:00:00:00", 0
    pull = 1001 / 1000 if "/" in rate else 1.0      # 23.976 / 29.97: timecode runs 0.1% slow
    frames = int((time_ref / sr) / pull * nominal + 1e-9)
    lead = int(round(time_ref - frames / nominal * pull * sr))
    return frames_to_tc(frames, nominal), max(0, lead)


def write_aaf(out_path, title, stems, sr, start_tc="01:00:00:00", fps=25.0, progress=lambda *a: None, lead=0):
    """stems: list of (track name, wav path). Returns out_path."""
    import aaf2
    rate, nominal = RATES.get(round(float(fps), 3), RATES[25.0])
    start = tc_to_frames(start_tc, nominal)
    tmp = tempfile.mkdtemp(prefix="untie_aaf_")
    try:
        # Pro Tools wants one mono file per track: split every stem into mono 24-bit WAVs
        tracks = []
        for name, path in stems:
            info = sf.info(path)
            if info.channels == 1:
                labels = [""]
            elif info.channels == 2:
                labels = [" L", " R"]
            else:
                labels = [" %d" % (c + 1) for c in range(info.channels)]
            monos = []
            for c, lab in enumerate(labels):
                p = os.path.join(tmp, "%d_%d.wav" % (len(tracks), c))
                monos.append((name + lab, p))
            outs = [sf.SoundFile(p, "w", sr, 1, "PCM_24", format="WAV") for _, p in monos]
            with sf.SoundFile(path) as src:
                for blk in src.blocks(blocksize=sr * 20, dtype="float32", always_2d=True):
                    for c, o in enumerate(outs):
                        o.write(blk[:, c])
            for o in outs:
                o.close()
            tracks.append((name, monos, info.frames))
            progress(5 + 25 * len(tracks) / len(stems), "Preparing AAF tracks", None)

        if os.path.exists(out_path):
            os.remove(out_path)
        n_tracks = sum(len(m) for _, m, _ in tracks)
        with aaf2.open(out_path, "w") as f:
            comp = f.create.CompositionMob(title)
            comp.usage = "Usage_TopLevel"
            f.content.mobs.append(comp)
            length = max(n for _, _, n in tracks)
            tc_slot = comp.create_timeline_slot(edit_rate=rate)
            tc_slot.name = "TC1"
            num, _, den = rate.partition("/")
            tc_len = int((length + lead) * int(num) / int(den or 1) / sr) + nominal
            tc = f.create.Timecode(nominal, drop=False, length=tc_len)
            tc.start = start
            tc_slot.segment = tc
            done = 0
            for name, monos, frames in tracks:
                for track_name, p in monos:
                    master = f.create.MasterMob(track_name)
                    f.content.mobs.append(master)
                    mslot = master.import_audio_essence(p, sr)
                    slot = comp.create_sound_slot(edit_rate=sr)
                    slot.name = track_name
                    if lead:
                        slot.segment.components.append(f.create.Filler("sound", int(lead)))
                    clip = master.create_source_clip(slot_id=mslot.slot_id, media_kind="sound")
                    slot.segment.components.append(clip)
                    done += 1
                    progress(30 + 70 * done / n_tracks, "Writing the AAF", None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path
