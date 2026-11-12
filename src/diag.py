"""Step 0: why is subject-mixed accuracy ~82% but LOSO ~35%?  Run:  python src/diag.py"""
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from common import *


def match_stat_columns(X, seg, thr=0.99):
    """Find which columns of X equal simple statistics of the (z-scored) segments.
    Returns {column: 'CH_stat'} for columns with |corr| > thr. Constant columns are reported separately."""
    cands, names = [], []
    for c in range(seg.shape[2]):
        s = seg[:, :, c].astype(np.float64)
        for nm, v in [("mean", s.mean(1)), ("std", s.std(1)), ("var", s.var(1)),
                      ("rms", np.sqrt((s ** 2).mean(1))), ("skew", skew(s, axis=1)),
                      ("kurt", kurtosis(s, axis=1))]:
            cands.append(v); names.append(f"{CH[c]}_{nm}")
    C = np.column_stack(cands); names = np.array(names)
    okc = C.std(0) > 1e-8
    C, names = C[:, okc], names[okc]
    ok = X.std(0) > 1e-8
    out = {}
    if C.shape[1] == 0 or not ok.any():
        return out
    Xn = (X[:, ok] - X[:, ok].mean(0)) / X[:, ok].std(0)
    Cn = (C - C.mean(0)) / C.std(0)
    R = np.abs(Xn.T @ Cn / len(X))
    cols = np.where(ok)[0]
    for i, j in enumerate(cols):
        if R[i].max() > thr:
            out[int(j)] = str(names[R[i].argmax()])
    return out


def main():
    X, y, groups = load_xy()
    print("X shape:", X.shape, "| subjects:", N_SUB, "| segments/subject:", SEG)
    print("class counts per subject:", np.bincount(y[::SEG]), "-> majority-class chance =",
          round(np.bincount(y[::SEG]).max() / N_SUB, 3))

    sd = X.std(0)
    const = np.where(sd < 1e-8)[0]
    print(f"\n[1] Constant features: {len(const)}  {const.tolist()[:40]}")

    eta = eta_sq_all(X, groups)
    print(f"\n[2] Median eta^2 (subject-identity share of variance): {np.nanmedian(eta):.3f}")
    order = np.argsort(-np.nan_to_num(eta))[:15]
    print("    Top-15 most subject-identifying columns:")
    for j in order:
        print(f"      col {j:3d}  (col//18={j//18}, col%18={j%18})  eta^2={eta[j]:.3f}")

    p = DATA / "X_processed.npy"
    if p.exists():
        seg = np.load(p)
        m = match_stat_columns(X, seg)
        print(f"\n[3] Columns that exactly match simple stats of data/X_processed.npy: {len(m)}")
        for j, nm in sorted(m.items()):
            print(f"      col {j:3d} = {nm:10s}  eta^2={eta[j]:.3f}")
        print("    mean/std/var/rms columns missing here AND counted constant above => features were computed"
              " on per-segment z-scored data (so 'AF3_Std' in SHAP must come from somewhere else).")
        print("    Compare with the SHAP table: if Std columns are NOT constant, they were NOT computed from"
              " per-segment z-scored data.")
    else:
        print("\n[3] data/X_processed.npy not found, skipping layout probe")

    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, groups):
        m = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.2, n_jobs=-1)
        m.fit(X[tr], groups[tr])
        accs.append((m.predict(X[te]) == groups[te]).mean())
    print(f"\n[4] Subject-ID accuracy (predict WHO, not load): {np.mean(accs):.3f} | chance = {1/N_SUB:.3f}")
    print("    far above chance => the features carry a subject fingerprint => mixed-split accuracy is inflated")


if __name__ == "__main__":
    main()
