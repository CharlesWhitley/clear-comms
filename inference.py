import torch
import torch.nn as nn
import soundfile as sf
import argparse
import os
import numpy as np

# training model 
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

# chunked inference
def separate_audio(input_path, checkpoint_path, output_path, device="cpu", chunk_size=441000):
    """
    chunk_size = number of samples per chunk (44100 = 1s at 44.1kHz, so 441000 = 10s)
    """
    # load audio
    audio, sr = sf.read(input_path, dtype="float32")
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]

    # load model
    model = SimpleSeparator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    output_chunks = []
    with torch.no_grad():
        for start in range(0, audio.shape[-1], chunk_size):
            end = min(start + chunk_size, audio.shape[-1])
            chunk = audio[:, :, start:end]

            pred = model(chunk)
            output_chunks.append(pred.squeeze().cpu().numpy())

    # concatenate all processed chunks
    pred_audio = np.concatenate(output_chunks)

    # save result
    sf.write(output_path, pred_audio, sr)
    print(f"✅ Saved separated gameplay audio to {output_path}")

# cli usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--input", type=str, required=True, help="Path to input .wav (casters + gameplay)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save separated gameplay")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--chunk_size", type=int, default=441000, help="Samples per chunk (default=10s at 44.1kHz)")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if "/" in args.output else None
    separate_audio(args.input, args.checkpoint, args.output, args.device, args.chunk_size)
