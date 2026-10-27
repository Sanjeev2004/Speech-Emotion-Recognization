# models/

Put exported files here (from `python -m ser.export_onnx ...` or the Kaggle zip `export/`):

- `<model>__<dataset>.onnx`  the network
- `<model>__<dataset>.json`  labels, input type, test scores
- `leaderboard.json`         model comparison shown on the website

The backend loads every `*.onnx` with a matching `.json` at startup.
Spectrogram models are 1-10 MB, so they can be committed to git and deployed on Render.
