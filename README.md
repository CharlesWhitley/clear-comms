Clear Comms is a project that uses deep learning to separate caster/announcer voices from gameplay audio in livestreams. 
Goal: give viewers the option to watch games with **pure gameplay audio** while muting commentary.

---

Features:
- AI Audio Separation: Model trained on synthetic datasets of gameplay + overlaid caster audio.   
- Custom Dataset Tools: Scripts to download VODs, mix in podcast/voice audio, and prepare train/validation splits.  
- Inference: Run trained model on new livestream audio to generate caster-free output.  
- Browser Extension (Prototype): Chrome extension with a simple enable/disable toggle. Injects into Twitch/YouTube tabs to eventually redirect audio through the model.  

---

Setup & Installation:
```bash
git clone https://github.com/<CharlesWhitley>/clear-comms.git
cd clear-comms-main

# create venv
python -m venv .venv
.venv\Scripts\activate   # windows

# install deps
pip install -r requirements.txt
```
1. Prepare dataset
```bash
python vod_download.py
python data_prep.py
python split_dataset.py
```

2. Train model
```bash
python train_model.py
```

3. Run Inference
```bash
python inference.py --input input.wav --checkpoint checkpoints/model_best.pth --output output.wav
```
