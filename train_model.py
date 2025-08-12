import sys
import os
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

import torch
from torchaudio.models import ConvTasNet
from data_loader import GamesData
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

model = ConvTasNet(num_sources=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

games = GamesData()
#Could use si sdr loss instead of L1 or L2
loss_fn = ScaleInvariantSignalDistortionRatio()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataloader = DataLoader(games, batch_size=2, shuffle=True, num_workers=4)
epochs = 10

for epoch in range(epochs):
    model.train()
    for mixed, clean in dataloader:
        mixed.to(device)
        clean.to(device)
        outputs = model(mixed).squeeze(1)

        loss = loss_fn(outputs, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('one batch done')
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

