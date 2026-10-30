# Resume and interview kit

Copy-paste material aligned with what this repository actually ships today.
Update numbers after you finish the full Kaggle GPU sweep.

## One-liner

Built an end-to-end Speech Emotion Recognition system: speaker-independent training and model comparison, ONNX export, FastAPI inference, and a React demo that classifies voice into six emotions.

## Resume bullets (pick 3–4)

- Designed a speaker-independent SER pipeline on RAVDESS and CREMA-D with shared NumPy feature extraction between training and serving (parity covered by tests).
- Compared classical ML and deep spectrogram models; deployed a BiLSTM + attention ONNX model (~0.63M params, ~2.5 MB) with **52.9% UAR** on held-out CREMA-D speakers (chance ≈ 16.7%).
- Served predictions with FastAPI + onnxruntime (no PyTorch in production) and a React/TypeScript UI for mic/upload analysis, leaderboard, confusion matrix, and local prediction history.
- Added CI (GitHub Actions), Docker Compose one-command demo, and documented live deployment on Vercel.

## Skills keywords

Python · PyTorch · ONNX · FastAPI · React · TypeScript · scikit-learn · speech ML · speaker-independent evaluation · CI/CD · Docker

## 60-second interview script

1. **Problem:** Detect emotion from *how* speech sounds, not the words.
2. **Data rigor:** Speaker-independent splits so the model cannot memorize voices.
3. **Models:** Classical baselines (SVM, RF) vs deep BiLSTM/CNN families; SSL (wav2vec2/HuBERT/WavLM) wired for GPU training.
4. **Production:** Export best small model to ONNX; browser sends 16 kHz WAV; API returns six probabilities in tens of ms once warm.
5. **Honest limit:** Current public numbers are CREMA-D CPU run; happy↔angry and fear↔sad confusions match known arousal patterns; full cross-dataset GPU sweep is the next experiment.

## Demo checklist before an interview

1. Open the live frontend; hit Analyze once so the free-tier backend wakes.
2. Record a short angry / sad / neutral sentence; show probability bars.
3. Open Model comparison → confusion matrix for BiLSTM.
4. Mention GitHub CI badge and `docker compose up --build` for local clone-and-run.

## What not to claim until artifacts exist

- “Benchmarked all 9 models across both corpora” — only after `reports/leaderboard.json` has those rows.
- “State-of-the-art UAR” — 52.9% is a solid engineering baseline, not SOTA for SSL SER.
