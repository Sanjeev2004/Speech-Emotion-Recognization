"""Train ONE model on ONE dataset and evaluate it on every dataset's test split.

Example:
    python -m ser.train --model cnn2d --dataset ravdess \
        --ravdess-dir data/ravdess --cremad-dir data/cremad

Output (runs/<model>__<dataset>/):
    result.json   metrics on each test set (same dataset AND cross-dataset)
    history.csv   loss / val score per epoch (deep models)
    best.pt       PyTorch weights   |  model.joblib  for scikit-learn models
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import build_manifest
from .features import build_cache
from .labels import EMOTIONS
from .metrics import compute_metrics
from .models import REGISTRY, build_model

N_CLASSES = len(EMOTIONS)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def split_indices(df: pd.DataFrame, dataset: str, split: str) -> np.ndarray:
    m = df["split"] == split
    if dataset != "combined":
        m &= df["dataset"] == dataset
    return np.where(m)[0]


def eval_targets(df: pd.DataFrame) -> list[str]:
    present = sorted(df["dataset"].unique())
    return present + (["combined"] if len(present) > 1 else [])


# ----------------------------------------------------------------------------- classical
def train_classical(args, df, feats, run_dir: Path) -> dict:
    import joblib
    X, y = np.asarray(feats["stats"]), df["label"].to_numpy()
    tr = np.concatenate([split_indices(df, args.dataset, "train"), split_indices(df, args.dataset, "val")])
    model = build_model(args.model, N_CLASSES, seed=args.seed)
    t0 = time.time()
    model.fit(X[tr], y[tr])
    train_time = time.time() - t0
    joblib.dump(model, run_dir / "model.joblib")

    results = {}
    for target in eval_targets(df):
        te = split_indices(df, target, "test")
        t1 = time.time()
        pred = model.predict(X[te])
        ms = (time.time() - t1) / max(1, len(te)) * 1000
        results[target] = compute_metrics(y[te], pred) | {"ms_per_clip": ms}
    return {"train_seconds": train_time, "n_params": None, "test": results, "best_epoch": None}


# ----------------------------------------------------------------------------- torch
def train_torch(args, df, feats, run_dir: Path) -> dict:
    import torch
    import torch.nn as nn

    from .models.deep import spec_augment

    spec = REGISTRY[args.model]
    hp = {**spec.defaults, **{k: v for k, v in
                              {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}.items()
                              if v is not None}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    X = feats[spec.input]
    y = df["label"].to_numpy()
    tr = split_indices(df, args.dataset, "train")
    va = split_indices(df, args.dataset, "val")

    model = build_model(args.model, N_CLASSES, freeze_backbone=args.freeze_backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {args.model} on {args.dataset}: {len(tr)} train / {len(va)} val clips, "
          f"{n_params / 1e6:.2f}M params, device={device}")

    # class weights: rare emotions count more in the loss
    counts = np.bincount(y[tr], minlength=N_CLASSES).astype(np.float32)
    weights = torch.tensor(counts.sum() / (N_CLASSES * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    if spec.kind == "ssl":
        groups = model.param_groups(hp["lr"], hp.get("head_lr", 1e-3))
        optim = torch.optim.AdamW(groups, weight_decay=1e-2)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)

    bs = hp["batch_size"]
    steps_per_epoch = math.ceil(len(tr) / bs)
    total = steps_per_epoch * hp["epochs"]
    warmup = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def batch(idx):
        xb = torch.from_numpy(np.asarray(X[np.sort(idx)], dtype=np.float32)).to(device)
        yb = torch.from_numpy(y[np.sort(idx)]).to(device)
        return xb, yb

    def augment(xb):
        if spec.input == "logmel":
            return spec_augment(xb)
        # waveform: random gain, light noise, random time shift
        gain = torch.empty(xb.size(0), 1, device=device).uniform_(0.6, 1.0)
        xb = xb * gain + 0.003 * torch.randn_like(xb) * torch.rand(xb.size(0), 1, device=device)
        shift = int(torch.randint(-4000, 4001, (1,)))
        return torch.roll(xb, shift, dims=1)

    @torch.no_grad()
    def predict(idx):
        model.eval()
        preds = []
        for i in range(0, len(idx), bs * 2):
            xb, _ = batch(idx[i:i + bs * 2])
            with torch.autocast(device.type, enabled=use_amp):
                preds.append(model(xb).argmax(1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=int)

    patience = args.patience or (4 if spec.kind == "ssl" else 12)
    best, best_epoch, patience_left, history, start_epoch, prev_time = -1.0, 0, patience, [], 1, 0.0
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        # continue an interrupted run (Kaggle/Colab timeouts, machine restarts)
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, best_epoch, patience_left = ck["best"], ck["best_epoch"], ck["patience_left"]
        history, start_epoch, prev_time = ck["history"], ck["epoch"] + 1, ck["train_time"]
        print(f"[train] resumed from epoch {ck['epoch']} (best val_uar {best:.3f})")
    t0 = time.time() - prev_time
    for epoch in range(start_epoch, hp["epochs"] + 1):
        if patience_left <= 0:
            break
        model.train()
        perm = np.random.permutation(tr)
        losses = []
        for i in range(0, len(perm), bs):
            xb, yb = batch(perm[i:i + bs])
            xb = augment(xb)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val = compute_metrics(y[np.sort(va)], predict(va))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_accuracy": val["accuracy"], "val_uar": val["uar"]})
        print(f"  epoch {epoch:3d}  loss {np.mean(losses):.4f}  val_acc {val['accuracy']:.3f}  "
              f"val_uar {val['uar']:.3f}")
        if val["uar"] > best:
            best, best_epoch, patience_left = val["uar"], epoch, patience
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            patience_left -= 1
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "best_epoch": best_epoch,
                    "patience_left": patience_left, "history": history,
                    "train_time": time.time() - t0}, last_path)
        if patience_left <= 0:
            print(f"  early stop (best val_uar {best:.3f} @ epoch {best_epoch})")
            break
    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    results = {}
    for target in eval_targets(df):
        te = split_indices(df, target, "test")
        t1 = time.time()
        pred = predict(te)
        ms = (time.time() - t1) / max(1, len(te)) * 1000
        results[target] = compute_metrics(y[np.sort(te)], pred) | {"ms_per_clip": ms}
    return {"train_seconds": train_time, "n_params": int(n_params), "test": results,
            "best_epoch": best_epoch, "best_val_uar": best, "hparams": hp, "device": str(device)}


# ----------------------------------------------------------------------------- main
def run(args) -> dict:
    set_seed(args.seed)
    df = build_manifest(args.ravdess_dir, args.cremad_dir, seed=args.seed)
    if args.dataset != "combined" and args.dataset not in set(df["dataset"]):
        raise SystemExit(f"Dataset '{args.dataset}' not found in the given folders.")
    feats = build_cache(df, args.cache_dir, n_jobs=args.n_jobs)
    run_dir = Path(args.out) / f"{args.model}__{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = REGISTRY[args.model]
    out = train_classical(args, df, feats, run_dir) if spec.kind == "classical" else \
        train_torch(args, df, feats, run_dir)
    result = {"model": args.model, "kind": spec.kind, "input": spec.input,
              "description": spec.description, "train_dataset": args.dataset,
              "emotions": EMOTIONS, "seed": args.seed,
              "split_sizes": df.groupby(["dataset", "split"]).size().unstack().to_dict(), **out}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    own = result["test"].get(args.dataset, {})
    print(f"[done] {args.model} / {args.dataset}: test acc {own.get('accuracy', 0):.3f}  "
          f"UAR {own.get('uar', 0):.3f}  -> {run_dir / 'result.json'}")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(REGISTRY))
    p.add_argument("--dataset", required=True, choices=["ravdess", "cremad", "combined"])
    p.add_argument("--ravdess-dir")
    p.add_argument("--cremad-dir")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--freeze-backbone", action="store_true", help="SSL models: train only the head")
    p.add_argument("--resume", action="store_true", help="continue from runs/<model>__<dataset>/last.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
