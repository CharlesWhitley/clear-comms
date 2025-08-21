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
.venv\Scripts\activate   # Windows

# install deps
pip install -r requirements.txt