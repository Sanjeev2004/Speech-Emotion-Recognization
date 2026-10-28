"""The backend must compute features exactly like training did."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_audio_features_identical():
    train_copy = ROOT / "ml" / "ser" / "audio_features.py"
    backend_copy = ROOT / "backend" / "app" / "audio_features.py"
    if not train_copy.exists():   # backend deployed alone
        return
    assert train_copy.read_bytes() == backend_copy.read_bytes(), \
        "Run: cp ml/ser/audio_features.py backend/app/audio_features.py"
