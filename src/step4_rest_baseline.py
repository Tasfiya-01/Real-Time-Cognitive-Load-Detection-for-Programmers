"""Step 4 (optional, needs STEW rest recordings): rest-baseline normalisation + rest-vs-task sanity check.

BEFORE RUNNING, edit the CONFIG block:
  * REST_GLOB   : where the 45 rest recordings are (each (19200, 14) text/CSV, same channel order as task)
  * SUBJECT ORDER must equal the order of your task data (sorted file names usually do; check the README)
  * ZSCORE_MODE : copy the convention from src/preprocess.py ("segment" or "recording")
Features are computed with YOUR extract_features_v3 so rest/task features are identical in definition.
"""
import glob
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from common import *
from loso_eval import make_xgb, table_row

# ---------------- CONFIG ----------------
REST_GLOB = str(DATA / "rest" / "*_lo.txt")     # e.g. data/rest/subject_01_lo.txt ... (CHECK the real names)
ZSCORE_MODE = "segment"                          # "segment" or "recording"  (match preprocess.py!)
FS, SEG_LEN, N_SEG = 128, 640, 30
# ----------------------------------------


def load_rest_raw():
    files = sorted(glob.glob(REST_GLOB))
    if len(files) != N_SUB:
        raise SystemExit(f"Found {len(files)} rest files with {REST_GLOB!r}, expected {N_SUB}. "
                         "Fix REST_GLOB / put the files there. Skipped subjects must match the task data "
                         "(thesis: subjects 5, 24, 42 had no ratings and were excluded).")
    return np.stack([np.loadtxt(f)[: N_SEG * SEG_LEN] for f in files])      # (45, 19200, 14)


def preprocess_like_thesis(raw):
    sos = butter(4, [0.5, 50], btype="band", fs=FS, output="sos")
    out = np.zeros((raw.shape[0] * N_SEG, SEG_LEN, raw.shape[2]))
    for s in range(raw.shape[0]):
        x = sosfiltfilt(sos, raw[s], axis=0)
        if ZSCORE_MODE == "recording":
            x = (x - x.mean(0)) / (x.std(0) + 1e-12)
        for k in range(N_SEG):
            seg = x[k * SEG_LEN:(k + 1) * SEG_LEN]
            if ZSCORE_MODE == "segment":
                seg = (seg - seg.mean(0)) / (seg.std(0) + 1e-12)
            out[s * N_SEG + k] = seg
    return out


def featurize(seg, extract_fn):
    return np.array([np.asarray(extract_fn(seg[i]), dtype=float).ravel() for i in range(len(seg))])


def loso_binary_segments(Xr, Xt, groups_s):
    """rest(0) vs task(1) LOSO: both classes exist inside every subject, so segment-level accuracy is meaningful."""
    X = np.vstack([Xr, Xt]); y = np.r_[np.zeros(len(Xr), int), np.ones(len(Xt), int)]
    g = np.r_[groups_s, groups_s]
    accs = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make_xgb(); m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        accs.append((m.predict(X[te]) == y[te]).mean())
    return np.array(accs)


def main(extract_fn=None, rest_raw=None):
    if extract_fn is None:
        from features_v3 import extract_features_v3 as extract_fn   # your own function
    X_task, y, groups = load_xy()
    rest = load_rest_raw() if rest_raw is None else rest_raw
    X_rest = featurize(preprocess_like_thesis(rest), extract_fn)
    assert X_rest.shape == X_task.shape, (X_rest.shape, X_task.shape)
    np.save(DATA / "X_rest_v3.npy", X_rest)

    base = X_rest.reshape(N_SUB, SEG, -1).mean(1, keepdims=True)        # each subject's own calm baseline
    X_delta = (X_task.reshape(N_SUB, SEG, -1) - base).reshape(N_SUB * SEG, -1)
    rows = [table_row("task features (as thesis)", X_task, y, groups),
            table_row("task - subject's REST mean", X_delta, y, groups),
            table_row("both: task + (task - rest)", np.hstack([X_task, X_delta]), y, groups)]
    out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(OUT / "rest_baseline.csv", index=False)

    a = loso_binary_segments(X_rest, X_task, groups)
    print(f"\nrest-vs-task LOSO (segment level, chance 50%): mean {100*a.mean():.1f}%  "
          f"(min {100*a.min():.0f}, max {100*a.max():.0f})")
    # NOTE: never normalise by the subject's own TASK mean -- the label is constant within a subject,
    # so that would erase the label signal itself.


if __name__ == "__main__":
    main()
