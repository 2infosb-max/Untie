"""Reading and writing audio files, keeping the Broadcast-WAV timecode stamp."""
import os
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import soundfile as sf

BEXT_FIXED = 602          # size of the fixed part of a 'bext' chunk


def read_bext_time_ref(path):
    """Return the BWF time reference (samples since midnight) or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[8:12] != b"WAVE" or head[:4] not in (b"RIFF", b"RF64", b"BW64"):
                return None
            while True:
                h = f.read(8)
                if len(h) < 8:
                    return None
                cid, size = h[:4], struct.unpack("<I", h[4:])[0]
                if cid == b"bext":
                    data = f.read(min(size, BEXT_FIXED))
                    if len(data) >= 346:
                        lo, hi = struct.unpack("<II", data[338:346])
                        return lo + (hi << 32)
                    return None
                if cid == b"data" and size == 0xFFFFFFFF:
                    return None
                f.seek(size + (size & 1), 1)
    except OSError:
        return None


def load(path):
    """Load any audio file as float32 (samples, channels). Returns (audio, rate, time_ref)."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x, sr, read_bext_time_ref(path)
    except Exception as first:
        # Formats libsndfile can't open (m4a, aac, mov, mp4…): let macOS decode them.
        if shutil.which("afconvert"):
            tmp = tempfile.mkdtemp(prefix="untie_")
            out = os.path.join(tmp, "decoded.wav")
            r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32", path, out],
                               capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(out):
                x, sr = sf.read(out, dtype="float32", always_2d=True)
                shutil.rmtree(tmp, ignore_errors=True)
                return x, sr, None
            shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"Can't read this file ({first}). Export it as a WAV or AIFF and try again.")


def _bext_chunk(time_ref, description):
    d = description.encode("ascii", "replace")[:256].ljust(256, b"\0")
    orig = b"Un-tie by Sound Brewery".ljust(32, b"\0")
    import datetime
    now = datetime.datetime.now()
    body = (d + orig + b"".ljust(32, b"\0") + now.strftime("%Y-%m-%d").encode() + now.strftime("%H:%M:%S").encode()
            + struct.pack("<II", time_ref & 0xFFFFFFFF, (time_ref >> 32) & 0xFFFFFFFF)
            + struct.pack("<H", 1) + b"\0" * 64 + b"\0" * 10 + b"\0" * 180)
    assert len(body) == BEXT_FIXED
    return b"bext" + struct.pack("<I", len(body)) + body


class WavWriter:
    """Streams a 24-bit PCM or 32-bit float WAV, with an optional BWF timecode stamp.
    Also re-opens an existing file to overwrite a stretch of samples in place."""

    def __init__(self, path, sr, channels, n_frames, fmt="pcm24", time_ref=None, description="", patch=False):
        self.path, self.ch, self.n = path, channels, n_frames
        self.float = fmt == "float32"
        self.bps = 4 if self.float else 3
        if patch:
            self.f = open(path, "r+b")
            self.data_off = self._find_data(self.f)
            return
        self.f = open(path, "wb")
        data_bytes = n_frames * channels * self.bps
        bext = _bext_chunk(time_ref, description) if time_ref is not None else b""
        if channels > 2 or self.float:
            # WAVE_FORMAT_EXTENSIBLE
            sub = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71" if self.float \
                else b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
            mask = {1: 0x4, 2: 0x3}.get(channels, 0)
            fmt_body = struct.pack("<HHIIHHHHI", 0xFFFE, channels, sr, sr * channels * self.bps,
                                   channels * self.bps, self.bps * 8, 22, self.bps * 8, mask) + sub
        else:
            fmt_body = struct.pack("<HHIIHH", 1, channels, sr, sr * channels * self.bps,
                                   channels * self.bps, self.bps * 8)
        fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
        riff_size = 4 + len(bext) + len(fmt_chunk) + 8 + data_bytes + (data_bytes & 1)
        if riff_size > 0xFFFFFFFF:
            raise ValueError("This file is too long for a WAV (over 4 GB per stem).")
        self.f.write(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + bext + fmt_chunk
                     + b"data" + struct.pack("<I", data_bytes))
        self.data_off = self.f.tell()
        self.f.truncate(self.data_off + data_bytes + (data_bytes & 1))

    @staticmethod
    def _find_data(f):
        f.seek(12)
        while True:
            h = f.read(8)
            if len(h) < 8:
                raise ValueError("bad wav")
            cid, size = h[:4], struct.unpack("<I", h[4:])[0]
            if cid == b"data":
                return f.tell()
            f.seek(size + (size & 1), 1)

    def write_at(self, start, x):
        """Write float samples x (frames, channels) starting at sample frame 'start'."""
        if start < 0:
            x = x[-start:]
            start = 0
        end = min(self.n, start + len(x))
        if end <= start:
            return
        x = x[:end - start]
        self.f.seek(self.data_off + start * self.ch * self.bps)
        self.f.write(encode(x, self.float))

    def close(self):
        self.f.close()


def encode(x, as_float):
    if as_float:
        return np.ascontiguousarray(x, dtype="<f4").tobytes()
    q = np.clip(np.round(x.astype(np.float64) * 8388608.0), -8388608, 8388607).astype("<i4")
    return q.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


def quantize24(x):
    """What a sample value becomes after being stored as 24-bit."""
    return np.clip(np.round(x.astype(np.float64) * 8388608.0), -8388608, 8388607) / 8388608.0
