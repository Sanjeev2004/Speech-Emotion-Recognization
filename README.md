# Speech Emotion Recognition

Detect emotion (neutral, happy, sad, angry, fear, disgust) from a voice clip. The project trains and compares **9 models** on **2 datasets**, and serves the best small model through a **FastAPI** backend and a **React** frontend. No Docker is needed anywhere.

**Live demo:** https://ser-frontend-beta.vercel.app (API: https://ser-backend-bice.vercel.app/docs)

```
                    Kaggle / Colab GPU                         Render (free)        Vercel (free)
 RAVDESS ─┐   ┌──────────────────────────────┐   ONNX file   ┌──────────────┐  JSON  ┌────────────┐
          ├──>│ ml/  features -> 9 models -> │ ────────────> │ backend/     │ <────> │ frontend/  │
 CREMA-D ─┘   │      leaderboard             │  leaderboard  │ FastAPI +    │  WAV   │ React +    │
              └──────────────────────────────┘               │ onnxruntime  │        │ Vite       │
                                                             └──────────────┘        └────────────┘
```

## Repository layout

| Folder | What it contains |
|---|---|
| `ml/` | Python package `ser`: dataset loaders, feature extraction, models, training, comparison, ONNX export |
| `ml/notebooks/train_on_kaggle.ipynb` | One notebook that trains everything on a free Kaggle GPU |
| `backend/` | FastAPI app: `/api/predict`, `/api/models`, `/api/results` |
| `frontend/` | React + TypeScript app: record or upload audio, see the prediction, compare models |
| `notebooks/legacy/` | The original TensorFlow CNN-LSTM notebook (kept for reference) |
| `render.yaml` | Render deployment config for the backend (native Python, no Docker) |

## Models compared

| Family | Model | Input | Idea |
|---|---|---|---|
| Classical ML | `svm` | 208 statistics per clip | RBF Support Vector Machine |
| Classical ML | `random_forest` | same | 500 decision trees voting |
| Deep learning | `bilstm` | log-mel spectrogram | LSTM reads frames in order, attention picks important moments |
| Deep learning | `cnn2d` | log-mel spectrogram | Spectrogram treated as an image |
| Deep learning | `cnn_lstm` | log-mel spectrogram | CNN for local patterns + BiLSTM over time (fixed version of the original) |
| Deep learning | `crnn_attention` | log-mel spectrogram | CNN + BiGRU + attention pooling |
| Self-supervised | `wav2vec2` | raw waveform | Pre-trained on 960 h of speech, fine-tuned |
| Self-supervised | `hubert` | raw waveform | Same size, different pre-training target |
| Self-supervised | `wavlm` | raw waveform | Pre-trained with noisy and overlapping speech |

Each model is trained 3 times: on RAVDESS, on CREMA-D, and on both combined. Every trained model is then tested on **every** test set, which gives two comparisons:

1. **Model vs model** on the same dataset.
2. **Dataset vs dataset** (cross-corpus): for example, train on RAVDESS and test on CREMA-D. This shows how well a model generalises to new recording conditions.

## Key terms (short notes)

- **Log-mel spectrogram**: a picture of the sound. x-axis = time (10 ms steps), y-axis = 64 frequency bands spaced like human hearing, brightness = loudness in dB.
- **MFCC**: a compressed version of the spectrogram (40 numbers per frame). Classic speech feature.
- **Speaker-independent split**: an actor is in train, validation **or** test, never in more than one. Without this, a model can recognise the voice instead of the emotion, and the test score looks better than reality.
- **Accuracy (WA)**: % of test clips predicted correctly.
- **UAR (unweighted average recall)**: recall of each emotion, averaged. The main metric in emotion research, because it does not reward ignoring rare classes.
- **ONNX**: a standard file format for trained networks. The backend runs it with `onnxruntime`, so the server does not need PyTorch (much smaller and lighter on RAM).
- **SpecAugment**: training trick that hides random time and frequency strips of the spectrogram so the model does not rely on any single detail.

## What changed from the original notebook

1. **Label bug fixed.** RAVDESS code `01` is *neutral*, `05` is *angry*. The old code mapped `01 -> Angry, 02 -> Happy ...` and labelled codes `07`/`08` as `unknown`, so the model learned wrong targets. This is the main reason for the ~22% test accuracy.
2. **Speaker-independent split** instead of a random split.
3. **CNN-LSTM reshape fixed.** Before, CNN output was flattened and reshaped to `(40, -1)`, mixing time and frequency. Now time stays the sequence axis.
4. Fixed 3 s clips at 16 kHz, silence trimming, class-weighted loss, SpecAugment, early stopping on validation UAR.
5. Feature extraction runs once and is cached, so all 27 runs reuse it.

## 1. Train and compare (Kaggle, free GPU)

1. Push this repo to GitHub.
2. On Kaggle, create a new notebook and upload `ml/notebooks/train_on_kaggle.ipynb` (File -> Import notebook).
3. Settings: Accelerator **GPU T4** and **Internet on**.
4. Add Input: `uwrfkaggler/ravdess-emotional-speech-audio` and `ejlok1/cremad`.
5. Run all. Spectrogram and classical models take about 1 hour in total. Transformer models take about 30-60 min per run. Runs that already finished are skipped, so if the session stops you can simply run again.
6. Download `ser_outputs.zip`. Copy everything inside `export/` into `backend/models/`.

Running locally (CPU) also works, but is only practical for the classical and spectrogram models:

```bash
cd ml
python -m venv .venv && .venv\Scripts\activate          # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m ser.train --model cnn2d --dataset ravdess --ravdess-dir path/to/ravdess --cremad-dir path/to/cremad
python -m ser.run_all --ravdess-dir ... --cremad-dir ... --models svm cnn2d cnn_lstm
python -m ser.compare                                    # -> reports/leaderboard.md, charts
python -m ser.export_onnx --run runs/cnn2d__combined --out ../backend/models
pytest -q                                                # smoke test on synthetic audio
```

## 2. Run the app locally

```bash
# terminal 1: backend
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000        # API docs at http://localhost:8000/docs

# terminal 2: frontend
cd frontend
npm install
npm run dev                                      # http://localhost:5173 (proxies /api to :8000)
```

## 3. Deploy (no Docker)

**Backend on Vercel (current live setup)**

1. Vercel -> Add New Project -> import the repo -> **Root Directory: `backend`**, framework **FastAPI**. Vercel installs `requirements.txt` and serves `index.py`.
2. Environment variables: `ALLOWED_ORIGIN_REGEX = https://.*\.vercel\.app`. Optional: `MODEL_BASE_URL` (a URL folder holding the `.onnx` files) if the model files are not part of the upload.
3. Check `https://<backend>.vercel.app/api/health`.

**Backend on Render (alternative)**

1. Make sure `backend/models/` contains the exported `.onnx`, `.json` and `leaderboard.json`, and is committed.
2. Render dashboard -> New -> Blueprint -> choose this repo. It reads `render.yaml` (Python runtime, `rootDir: backend`).
3. After the first deploy, open `https://<your-service>.onrender.com/api/health`.
4. Free instances sleep after 15 minutes of no traffic. The first request after that takes 30-60 s; the frontend shows a "waking up" notice.

**Frontend on Vercel**

1. Vercel -> Add New Project -> import the repo -> **Root Directory: `frontend`** (framework preset Vite is detected).
2. Environment variable: `VITE_API_URL = https://<your-service>.onrender.com`.
3. Deploy. `render.yaml` already allows any `*.vercel.app` origin through CORS. For a custom domain, add it to `ALLOWED_ORIGINS` on Render.

Why only spectrogram models are deployed: the Render free tier has 512 MB RAM. A spectrogram model is 1-10 MB as ONNX; wav2vec2-class models are about 360 MB and need more memory. They stay in the comparison and can be deployed on a bigger instance later.

## Results

First run: CREMA-D only, trained on CPU (no GPU), speaker-independent test set of 14 unseen actors (1,147 clips). Chance level is 16.7%.

| Rank | Model | Accuracy | UAR | Macro-F1 | Params |
|---|---|---|---|---|---|
| 1 | bilstm | 0.526 | 0.529 | 0.522 | 0.63M |
| 2 | svm | 0.499 | 0.498 | 0.496 | - |
| 3 | random_forest | 0.477 | 0.479 | 0.457 | - |

The deployed model is `bilstm__cremad`. The remaining models (CNN, CNN-LSTM, CRNN, wav2vec2, HuBERT, WavLM) and RAVDESS / combined / cross-dataset runs need a GPU: run `ml/notebooks/train_on_kaggle.ipynb`, then replace `reports/` and `backend/models/` with its output. The website's *Model comparison* tab reads `backend/models/leaderboard.json`, so it updates automatically.

## Datasets and licence notes

- RAVDESS: Livingstone and Russo (2018), CC BY-NC-SA 4.0.
- CREMA-D: Cao et al. (2014), Open Database License.

Both are for non-commercial research use. The audio itself is not stored in this repo.
