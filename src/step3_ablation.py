"""Step 3b: does adding the new features help?  python src/step3_ablation.py"""
import numpy as np, pandas as pd
from common import *
from loso_eval import table_row


def main():
    X, y, groups = load_xy()
    H = np.load(DATA / "X_new_hjorth_dwt.npy")
    A = np.load(DATA / "X_new_asym.npy")
    sets = {
        "base (v3)": X,
        "base + asymmetry": np.hstack([X, A]),
        "base + Hjorth/DWT": np.hstack([X, H]),
        "base + all new": np.hstack([X, A, H]),
        "new features only": np.hstack([A, H]),
    }
    out = pd.DataFrame([table_row(n, Xs, y, groups) for n, Xs in sets.items()])
    print(out.to_string(index=False))
    out.to_csv(OUT / "ablation_new_features.csv", index=False)


if __name__ == "__main__":
    main()
