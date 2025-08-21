import soundfile as sf
import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SISDR

class GamesDataset(Dataset):
    def __init__(self, mixed_dir, clean_dir, target_len=160000):  # default length
        self.mixed_files = sorted(os.listdir(mixed_dir))
        self.clean_files = sorted(os.listdir(clean_dir))
        self.mixed_dir = mixed_dir
        self.clean_dir = clean_dir
        self.target_len = target_len  # stored as attribute

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, idx):
        mixed_path = os.path.join(self.mixed_dir, self.mixed_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        mixed, sr = sf.read(mixed_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")

        mixed = torch.tensor(mixed).unsqueeze(0)
        clean = torch.tensor(clean).unsqueeze(0)

        # pad/trim
        if mixed.size(1) > self.target_len:
            mixed = mixed[:, :self.target_len]
            clean = clean[:, :self.target_len]
        elif mixed.size(1) < self.target_len:
            pad_amt = self.target_len - mixed.size(1)
            mixed = torch.nn.functional.pad(mixed, (0, pad_amt))
            clean = torch.nn.functional.pad(clean, (0, pad_amt))

        return mixed, clean

# simple model 
class SimpleSeparator(nn.Module):
    def __init__(self):
        super(SimpleSeparator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        return self.net(x)

# training loop
def train_model(train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model = SimpleSeparator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    sisdr = SISDR()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for mixed, clean in train_loader:
            mixed, clean = mixed.to(device), clean.to(device)

            # expects [batch, channels, time]
            mixed = mixed.squeeze(1).unsqueeze(1)
            clean = clean.squeeze(1).unsqueeze(1)

            pred = model(mixed)
            loss = criterion(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # to validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixed, clean in val_loader:
                mixed, clean = mixed.to(device), clean.to(device)
                mixed = mixed.squeeze(1).unsqueeze(1)
                clean = clean.squeeze(1).unsqueeze(1)

                pred = model(mixed)
                loss = criterion(pred, clean)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # save checkpoint
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    return model

# main
if __name__ == "__main__":
    BATCH_SIZE = 4
    EPOCHS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = GamesDataset("dataset/train/mixed", "dataset/train/gameplay_only")
    val_dataset = GamesDataset("dataset/val/mixed", "dataset/val/gameplay_only")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    os.makedirs("checkpoints", exist_ok=True)
    model = train_model(train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
