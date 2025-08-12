import yt_dlp
from pydub import AudioSegment
import os
NUM_VIDEO = 5
THIRTY_SEC = 30000
#Default sample rate is 44100hz
"""""
URLS_podcast = ['https://www.youtube.com/playlist?list=PLtiWkKVZkCXVu_3pkKsQviOZ7r_9b39JN']

ydl_opts_caster = {
    'format': 'm4a',
    # 'ffmpeg_location': 'C:/ffmpeg-master-latest-win64-gpl-shared/bin',
    # # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    # 'postprocessors': [{  # Extract audio using ffmpeg
    #     'key': 'FFmpegExtractAudio',
    #     'preferredcodec': 'wav',
    # }],
    'playliststart' : '1',
    'playlistend' : NUM_VIDEO,
    #'matchtitle' : r'^(?=.*\bvs\b)(?=.*\bGame\b).*',
    'paths' : {'home' : 'audios_podcast'}
}

URLS_no_caster = ['https://www.youtube.com/playlist?list=PLPUygacvheSPDBGj-gbgsYw8BsP7ETU_V']

ydl_opts_nocaster = {
    'format': 'm4a',
    # 'ffmpeg_location': 'C:/ffmpeg-master-latest-win64-gpl-shared/bin',
    # # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    # 'postprocessors': [{  # Extract audio using ffmpeg
    #     'key': 'FFmpegExtractAudio',
    #     'preferredcodec': 'wav',
    # }],
    'playliststart' : '1',
    'playlistend' : NUM_VIDEO,
    'paths' : {'home' : 'audios_nocaster'}
}
with yt_dlp.YoutubeDL(ydl_opts_caster) as ydl:
    error_code = ydl.download(URLS_podcast)

with yt_dlp.YoutubeDL(ydl_opts_nocaster) as ydl:
    error_code = ydl.download(URLS_no_caster)
"""

directory_podcast = os.listdir('audios_podcast')
directory_nocaster = os.listdir('audios_nocaster')

for i in range(NUM_VIDEO):
    podcast_file = os.path.join('audios_podcast', directory_podcast[i+1])
    nocaster_file = os.path.join('audios_nocaster', directory_nocaster[i+1])

    podcast = AudioSegment.from_file(podcast_file)
    nocaster = AudioSegment.from_file(nocaster_file)

    # Remove first 30s of podcast and overlay
    overlapped = nocaster.overlay(podcast[THIRTY_SEC:])
    overlapped.export(f'audios_caster/podcast+nocaster{i}.wav', format='wav')
