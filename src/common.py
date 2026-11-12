"""Shared helpers. Run every script from anywhere: paths are relative to the project root."""
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

N_SUB, SEG, SEED = 45, 30, 42
N_EST = int(os.environ.get("N_EST", 200))          # set N_EST=60 for a quick test run
FEATURE_FILE = "X_features_v3_fixed.npy"                 # final 252-feature matrix used by train_xgb.py
CH = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def load_xy(name=FEATURE_FILE):
    X = np.load(DATA / name)
    _, y = np.unique(np.load(DATA / "y_labels.npy"), return_inverse=True)
    groups = np.repeat(np.arange(N_SUB), SEG)        # assumes segments are ordered subject by subject
    assert X.shape[0] == len(y) == N_SUB * SEG, f"unexpected shapes: X{X.shape} y{y.shape}"
    for g in range(N_SUB):
        assert len(np.unique(y[groups == g])) == 1, (
            f"subject {g} has mixed labels -> segment order is not subject-major; fix 'groups'")
    return X, y, groups


def eta_sq_all(X, groups):
    """Share of each feature's variance explained by subject identity (1 = pure subject fingerprint)."""
    gm = X.mean(0)
    means = np.stack([X[groups == g].mean(0) for g in range(N_SUB)])
    ssb = SEG * ((means - gm) ** 2).sum(0)
    sst = ((X - gm) ** 2).sum(0)
    return np.where(sst > 1e-12, ssb / np.maximum(sst, 1e-12), np.nan)
