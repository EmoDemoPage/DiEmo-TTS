# DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech <br><sub>[Demo page](https://emodemopage.github.io/DiEmo-TTS-Demo/)</sub>

## Training Procedure

### Environments
```
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # install force alignment tools
```

### 1. Preprocess data
- We use ESD database, which is an emotional speech database that can be downloaded here: [ESD](https://hltsingapore.github.io/ESD/). 
- We follow the preprocessing methods referenced from [NATSpeech](https://github.com/NATSpeech/NATSpeech).

### 2. Clustering and Matching of Emotion
- We follow the methodology of Emotional Attribute Prediction and Categorical Emotion Recognition models referenced from [Speech Emotion Recognition](https://github.com/msplabresearch/MSP-Podcast_Challenge_IS2025).
- The related process can be checked through the following .sh script.
```
sh ./DiEmo_cluster_analyzing/Analyzing.sh
```

### 3. Training TTS module and Inference  
```bash
sh DiEmoTTS.sh
```
