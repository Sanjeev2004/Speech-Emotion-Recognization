"""Train every (model x dataset) combination, then build the comparison report.

    python -m ser.run_all --ravdess-dir data/ravdess --cremad-dir data/cremad
    python -m ser.run_all ... --models svm cnn2d cnn_lstm --datasets ravdess

Finished runs (result.json exists) are skipped, so you can stop and resume
(useful on Kaggle/Colab where sessions time out).
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from . import compare, train
from .models import REGISTRY


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(REGISTRY))
    p.add_argument("--datasets", nargs="+", default=["ravdess", "cremad", "combined"])
    p.add_argument("--ravdess-dir")
    p.add_argument("--cremad-dir")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--epochs", type=int, help="override epochs for every model (quick tests)")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--force", action="store_true", help="re-train even if result.json exists")
    a = p.parse_args()

    for dataset in a.datasets:
        for model in a.models:
            if (Path(a.out) / f"{model}__{dataset}" / "result.json").exists() and not a.force:
                print(f"[skip] {model} / {dataset} already done")
                continue
            argv = ["--model", model, "--dataset", dataset, "--cache-dir", a.cache_dir, "--out", a.out,
                    "--resume"]   # picks up half-finished runs after a session timeout
            if a.ravdess_dir:
                argv += ["--ravdess-dir", a.ravdess_dir]
            if a.cremad_dir:
                argv += ["--cremad-dir", a.cremad_dir]
            if a.epochs:
                argv += ["--epochs", str(a.epochs)]
            if a.freeze_backbone:
                argv += ["--freeze-backbone"]
            try:
                train.run(train.parser().parse_args(argv))
            except Exception:  # one failing model should not kill the whole sweep
                traceback.print_exc()
    compare.build_report(a.out, a.report_dir)


if __name__ == "__main__":
    main()
