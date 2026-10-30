# Claude Handoff — SER Kaggle T4 training (Sanjeev)

Ye note Claude ke liye hai taaki bilkul yahi workflow continue ho, kuch break na ho.

**Repo:** `C:\Users\sanje\OneDrive\Desktop\SER\Speech-Emotion-Recognization`  
**Branch:** `full-project` (GitHub: `Sanjeev2004/Speech-Emotion-Recognization`)  
**Kaggle user:** `salangsanjeev`  
**Live notebook URL:** https://www.kaggle.com/code/salangsanjeev/ser-train-all-models-t4

---

## 1) Ab tak kya ho chuka hai

### Resume / repo polish (already on `full-project`, pushed)
- MIT LICENSE, CI, Docker Compose, prediction history, confusion matrix UI, deploy hardening, `docs/RESUME.md`
- Commit example: `f9daf03` — resume-ready polish

### Kaggle auth (DONE — dubara mat setup karo)
- User ne legacy tokens expire kiye, naya token banaya (name ~ “one”)
- File path user ne diya: `C:\Users\sanje\Downloads\kaggle (1).json`
- Install location (Windows): `C:\Users\sanje\.kaggle\kaggle.json`
- Pehle ACL lock ki wajah se overwrite fail hua tha; unlock karke naya token copy ho gaya
- Auth verify: `python -m kaggle datasets list -s cremad --max-size 1` → **AUTH_OK**
- **kaggle.json contents kabhi print/log mat karna**

### Kaggle T4 job (SUBMITTED)
- Kernel push folder: `ml/kaggle_kernel_push/`
  - `train_on_kaggle.ipynb` (clone line patched to `-b full-project`)
  - `kernel-metadata.json` — GPU T4, internet on, datasets attached:
    - `uwrfkaggler/ravdess-emotional-speech-audio`
    - `ejlok1/cremad`
- Push command used: `python -m kaggle kernels push -p ml/kaggle_kernel_push --accelerator NvidiaTeslaT4`
- Pehle status **RUNNING** tha
- **Handoff likhte waqt status: `KernelWorkerStatus.ERROR`** — training finish nahi hui; Claude pehle isko diagnose/fix kare
- Local `ml/data/` **download nahi hua** (user ne baad mein bola: submit ke baad STOP — datasets local mat laao). Data Kaggle kernel inputs pe hai.

### Metrics abhi repo mein
Sirf pehle wale CPU CREMA-D results:
- bilstm / svm / random_forest in `reports/leaderboard.*`
- Baaki 6 models (cnn2d, cnn_lstm, crnn_attention, wav2vec2, hubert, wavlm) = **code only, no numbers yet**

---

## 2) Ab kya pending hai (original end-to-end goal)

1. **Kaggle run ko successful banao** (ab ERROR hai → logs dekho, fix, re-push / resume)
2. Sab **9 models** train + eval (CREMA-D; RAVDESS/combined jab feasible):
   - svm, random_forest, bilstm, cnn2d, cnn_lstm, crnn_attention, wav2vec2, hubert, wavlm
3. Metrics collect: accuracy, UAR, macro-F1, confusion matrix (+ precision/recall jahan pipeline de)
4. Download `ser_outputs.zip` / `reports/` + ONNX `export/` from kernel output
5. Copy into repo:
   - `reports/` ← leaderboard + charts
   - `backend/models/` ← ONNX + json + leaderboard.json
6. README regenerate: full results + **model selection guide** (kis model ko pick/skip, stats ke saath)
7. Deploy surface: best deployable model(s) ONNX; frontend comparison/selection pe naye numbers dikhein
8. Commit/push jab Sanjeev bole

**Abhi mat karo (user ne explicitly pehle rukne ko kaha tha):** local dataset re-download, README rewrite, deploy — **pehle successful Kaggle training + results lao.**

---

## 3) Exact continue steps (Claude)

### Auth check only (naya token mat banao)
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
python -m kaggle datasets list -s cremad --max-size 1
python -m kaggle kernels status salangsanjeev/ser-train-all-models-t4
```
Token path: `%USERPROFILE%\.kaggle\kaggle.json` — content echo mat karna.

### ERROR diagnose
```powershell
# Notebook page / logs on Kaggle UI, OR:
python -m kaggle kernels output salangsanjeev/ser-train-all-models-t4 -p reports/kaggle_kernel_output
```
Likely early fail: git clone / pip / path. Notebook source of truth:
- Repo notebook: `ml/notebooks/train_on_kaggle.ipynb`
- Push copy: `ml/kaggle_kernel_push/train_on_kaggle.ipynb` (must use `-b full-project`)

### Fix + re-push (sirf jab ERROR / incomplete)
1. Fix notebook cell(s) in `ml/kaggle_kernel_push/` (aur ideally sync `ml/notebooks/train_on_kaggle.ipynb`)
2. Metadata: `enable_gpu=true`, `machine_shape=NvidiaTeslaT4`, `enable_internet=true`, dono dataset_sources
3. UTF-8 **without BOM** for `kernel-metadata.json` (PowerShell `Set-Content` BOM se pehle push fail hua tha)
4. `python -m kaggle kernels push -p ml/kaggle_kernel_push --accelerator NvidiaTeslaT4`
5. Poll: `python -m kaggle kernels status salangsanjeev/ser-train-all-models-t4` until COMPLETE
6. Tab output download → `reports/` + `backend/models/`

### Training command (notebook ke andar already)
```text
python -m ser.run_all ... --models svm random_forest bilstm cnn2d cnn_lstm crnn_attention
python -m ser.run_all ... --models wav2vec2 hubert wavlm
python -m ser.compare ...
python -m ser.export_onnx ...
```
Finished runs skip hote hain unless `--force`. Fresh `/kaggle/working` pe sab chalna chahiye.

### Colab backup (optional, secondary)
- `ml/notebooks/train_on_colab.ipynb` exists
- **Recommendation earlier:** is project ke liye **Kaggle T4 better** (datasets native inputs)

---

## 4) DO NOT BREAK

- **Naya kaggle.json mat generate / expire mat karo** — current token already works
- Auth “from scratch” mat redo — sirf verify
- **Local RAVDESS/CREMA-D dubara download mat karo** unless Sanjeev explicitly bole (kernel pe datasets already attached)
- Successful RUNNING job ko casually cancel mat karo; abhi **ERROR** hai isliye fix+repush OK
- Secrets (token/key) chat, logs, commits mein mat daalna; `.kaggle/` git mein mat add karna
- Pehle **training success + metrics** complete karo; logging/token-saving refactors, README rewrite, deploy — baad mein, results ke baad
- Chat light rakhna ok, lekin pehle task finish priority; alag “token reduction” refactor abhi mat shuru

---

## 5) Quick truth table

| Item | Status |
|------|--------|
| Kaggle CLI auth | OK |
| Kernel pushed T4 | Yes (v1) |
| Kernel status (handoff time) | **ERROR** — needs fix |
| Local datasets | Not downloaded (by design) |
| New 9-model metrics in repo | Not yet |
| README selection guide / new ONNX deploy | Not yet |

Sanjeev se bole bina: force-push main, token rotate, ya unrelated refactors mat karna.
