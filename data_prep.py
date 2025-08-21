import os
import random
from pathlib import Path
from pydub import AudioSegment
import math

# config
GAMEPLAY_DIR = "audios_nocaster"
CASTER_DIR = "audios_caster"
PODCAST_DIR = "audios_podcast"  
OUT_MIX_DIR = "dataset/mixed"
OUT_CLEAN_DIR = "dataset/gameplay_only"
SAMPLE_RATE = 44100
TARGET_DURATION_MS = 30 * 1000  # 30s segments
MIN_GAIN_DB = -6  # reduce caster loudness a bit
MAX_GAIN_DB = 0

def ensure_dirs():
    os.makedirs(OUT_MIX_DIR, exist_ok=True)
    os.makedirs(OUT_CLEAN_DIR, exist_ok=True)

def to_wav_mono(aud: AudioSegment):
    aud = aud.set_frame_rate(SAMPLE_RATE)
    aud = aud.set_channels(1)
    aud = aud.set_sample_width(2)  # 16-bit
    return aud

def load_audio_files(folder):
    files = []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg']:
            files.append(p)
    return files

def make_segments(audio, duration_ms):
    # split audio into contiguous segments of duration_ms 
    segments = []
    length = len(audio)
    for start in range(0, length, duration_ms):
        seg = audio[start:start + duration_ms]
        if len(seg) < 1000:  # ignore if too small
            continue
        segments.append(seg)
    return segments

def create_pairs():
    ensure_dirs()
    gameplay_files = load_audio_files(GAMEPLAY_DIR)
    caster_files = load_audio_files(CASTER_DIR) + load_audio_files(PODCAST_DIR)
    if not gameplay_files:
        print("No gameplay files found in", GAMEPLAY_DIR)
        return
    if not caster_files:
        print("Warning: no caster files found; produced dataset will be identical to gameplay (not ideal)")

    pair_idx = 0
    for gp_path in gameplay_files:
        gp = AudioSegment.from_file(gp_path)
        gp = to_wav_mono(gp)

        # split into N segments
        segments = make_segments(gp, TARGET_DURATION_MS)
        for seg_idx, seg in enumerate(segments):
            # randomly pick a caster clip segment
            mix = seg
            if caster_files:
                caster_path = random.choice(caster_files)
                caster_a = AudioSegment.from_file(caster_path)
                caster_a = to_wav_mono(caster_a)
                caster_segments = make_segments(caster_a, TARGET_DURATION_MS)
                if caster_segments:
                    caster_seg = random.choice(caster_segments)

                    # randomly choose offset to overlay caster audio 
                    offset_ms = random.randint(0, max(0, len(seg) - 1000))
                    # randomly adjust gain so caster isn't unnaturally loud
                    gain_db = random.uniform(MIN_GAIN_DB, MAX_GAIN_DB)
                    caster_seg = caster_seg + gain_db

                    # overlay
                    mix = seg.overlay(caster_seg, position=offset_ms)

            # export both mixed and clean
            fname = f"clip_{pair_idx:06d}.wav"
            mix.export(os.path.join(OUT_MIX_DIR, fname), format="wav")
            seg.export(os.path.join(OUT_CLEAN_DIR, fname), format="wav")

            pair_idx += 1
            if pair_idx % 50 == 0:
                print("Created pairs:", pair_idx)
    print("Done. Created", pair_idx, "pairs in", OUT_MIX_DIR)

if __name__ == "__main__":
    create_pairs()
