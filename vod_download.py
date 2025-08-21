import os
import subprocess
from pathlib import Path

# config
URLS_FILE = "vod_list.txt"       
OUT_DIR = "audios_caster"   
SAMPLE_RATE = 44100             

Path(OUT_DIR).mkdir(exist_ok=True)

def download_audio(url, out_dir):
    # download
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",  # best quality
        "-o", f"{out_dir}/%(title)s.%(ext)s",
        url
    ]
    subprocess.run(cmd, check=True)

    # convert all wavs in folder to mono w fixed sample rate
    for wav_file in Path(out_dir).glob("*.wav"):
        tmp_file = wav_file.with_suffix(".tmp.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_file),
            "-ac", "1",                # mono
            "-ar", str(SAMPLE_RATE),   # sample rate
            str(tmp_file)
        ]
        subprocess.run(cmd, check=True)
        wav_file.unlink()  # remove original
        tmp_file.rename(wav_file)

def main():
    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        try:
            print(f"Downloading {url}...")
            download_audio(url, OUT_DIR)
            print(f"Finished {url}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    main()
