import torchaudio.transforms as T
SAMPLE_RATE = 44100
"""""
#Splits a waveform into chunks, has overlap too
#I'm trying to get 128 x 128 for the mel spectrogram but splitting waveform needs to be different?
#Not good for now but can fix later
#Unfold can also be batched can switch if not fast enough
#Good for spectrogram but probably isn't needed for waveforms
def split_waveform(waveform, window_size_ms=200, hop_length=50, overlap_ratio=0.25):
    num_frames = 128
    chunk_size = (num_frames - 1) * hop_length + window_size_ms  #128 time frames

    step_size = int(chunk_size * (1 - overlap_ratio))  #Control overlap with ratio we want
    chunks = waveform.T.unfold(dimension=0, size=chunk_size, step=step_size)
    return chunks.squeeze()  #tuple of (# of channels(should always be 1), # windows, size of window)

"""

def split_waveform(waveform, overlap_ratio = 0.1):
    chunk_size = SAMPLE_RATE *2 #2 second chunk
    step_size = int(chunk_size / (1 - overlap_ratio))
    chunks = waveform.unfold(dimension=0, size=chunk_size, step=step_size)
    return chunks.squeeze()

#https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
#Used with CNNs well apparently
#n_fft is the FFT window size increasing is longer computation, but better frequency res
#overlap length is amount of overlap of each chunk
#win_length means windows become taller, but time resolution shorter
def waveform_to_spectrogram(waveform, sample_rate, n_fft=2048, win_length=400, overlap_length=100):
    mel = T.MelSpectrogram(sample_rate, n_fft=n_fft, win_length=win_length, hop_length=overlap_length)
    return mel(waveform) #Should return a 1 x 128 x 128 tensor
