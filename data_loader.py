import sys
import os
from os import walk
# Manually call the _init_dll_path method to ensure that the system path is searched for FFMPEG.
# Calling torchaudio._extension.utils._init_dll_path does not work because it is initializing the torchadio module prematurely or something.
# See: https://github.com/pytorch/audio/issues/3789
if sys.platform == "win32":
    #print("Initializing DLL path for Windows")
    for path in os.environ.get("PATH", "").split(";"):
        if os.path.exists(path): 
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

from torch.utils.data import Dataset
import torchaudio
import torch
import numpy as np
from waveform_transforms import split_waveform, waveform_to_spectrogram
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 44100
#Custom dataloader that takes in the audio folder that holds the game audio
#Which then becomes loaded into torch waveforms accessable in an array
#We then create chunks of the audio with overlap(might need to tinker with this, one thing is that
#the Mel spectrogram stuff also makes them with overlap so we might be double dipping which is bad but have to test)
#The chunks are then made into spectrograms for the neural net to process
class GamesData(Dataset):
    def __init__(self):
        self.paths = ["audios_caster", "audios_nocaster"]
        self.chunks_caster = []
        self.chunks_nocaster = []
        for path in self.paths:
            for (dirpath, dirnames, filenames) in walk(path):
                for filename in filenames:
                    if filename!=".gitkeep":
                        filepath = os.path.join(dirpath, filename)
                        waveform, sample_rate = torchaudio.load(filepath, backend="ffmpeg")
                        waveform = waveform[0]
                        chunks = self.get_chunks(waveform)
                        if "podcast" in filepath:
                            self.chunks_caster.append(chunks)
                        else: 
                            self.chunks_nocaster.append(chunks)
                break  
        self.chunks_caster = torch.cat(self.chunks_caster, dim=0) #dim is total chunks x chunk size(see sample rate)
        self.chunks_nocaster = torch.cat(self.chunks_nocaster, dim=0)
        

    def __len__(self):
        return len(self.chunks_caster)
    
    def __getitem__(self, index):
        return self.chunks_caster[index], self.chunks_nocaster[index]
    
    #Creates chunks from the given waveform
    def get_chunks(self, waveform):
        return split_waveform(waveform).squeeze()
    
    #Given chunks creates mel spectrogram
    def get_mel(self, chunks):
        return torch.from_numpy(np.array([waveform_to_spectrogram(chunk, SAMPLE_RATE) for chunk in chunks]))

#Example usage
#example = GamesData()
#dataloader = DataLoader(example, batch_size=4, shuffle=True)