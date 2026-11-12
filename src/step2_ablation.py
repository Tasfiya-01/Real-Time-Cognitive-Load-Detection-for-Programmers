"""Step 2: which features are subject fingerprints?  Layout-free (uses eta^2) + auto-matched amplitude columns."""
import numpy as np, pandas as pd
from common import *
from loso_eval import table_row
from diag import match_stat_columns


def main():
    X, y, groups = load_xy()
    eta = eta_sq_all(X, groups)
    order = np.argsort(-np.nan_to_num(eta))           # most subject-identifying first
    rows = [table_row("all features", X, y, groups)]

    p = DATA / "X_processed.npy"
    if p.exists():
        m = match_stat_columns(X, np.load(p))
        amp = [j for j, nm in m.items() if nm.split("_")[1] in ("mean", "std", "var", "rms")]
        if amp:
            keep = np.setdiff1d(np.arange(X.shape[1]), amp)
            rows.append(table_row(f"no amplitude cols ({len(amp)})", X[:, keep], y, groups))
            rows.append(table_row(f"ONLY amplitude cols ({len(amp)})", X[:, amp], y, groups))

    for k in (10, 30, 60):
        rows.append(table_row(f"drop top-{k} subject-identifying", X[:, order[k:]], y, groups))
    for k in (10, 30):
        rows.append(table_row(f"ONLY top-{k} subject-identifying", X[:, order[:k]], y, groups))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT / "ablation_fingerprint.csv", index=False)


if __name__ == "__main__":
    main()
