"""Subject-level LOSO evaluation + subject-mixed evaluation + paired tests."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from common import *


def make_xgb():
    return XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1,
                         eval_metric="mlogloss", random_state=SEED, n_jobs=-1)


def run_loso(X, y, groups, kind="xgb", verbose=True):
    # Drop (near-)constant columns so StandardScaler does not blow up ~1e-10 noise.
    # Label-free, so no leakage.
    keep = X.std(axis=0) > 1e-6
    if verbose:
        print(f"[{kind}] kept {keep.sum()} of {X.shape[1]} features")
    X = X[:, keep]

    k = int(y.max()) + 1
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if kind == "xgb":
            m = make_xgb()
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        else:  # scaler is fitted on training subjects only
            m = make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=SEED))
            m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        row = dict(
            subject=int(groups[te][0]),
            true=int(y[te][0]),
            seg_acc=float((p == y[te]).mean()),
            vote=int(np.bincount(p, minlength=k).argmax()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, name, n_boot=5000):
    ok = (df.vote == df.true).to_numpy()
    rng = np.random.default_rng(SEED)
    b = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(model=name, subject_acc=ok.mean(), ci95_lo=lo, ci95_hi=hi,
                mean_segment_acc=df.seg_acc.mean(),
                balanced_acc=balanced_accuracy_score(df.true, df.vote),
                macro_f1=f1_score(df.true, df.vote, average="macro"))


def eval_mixed(X, y, n_splits=5):
    """Subject-mixed stratified 80/20 (the thesis headline protocol), repeated over seeds."""
    a = []
    for tr, te in StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=SEED).split(X, y):
        m = make_xgb()
        m.fit(X[tr], y[tr])
        a.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(a)), float(np.std(a))


def table_row(name, Xs, y, groups):
    mm, ms = eval_mixed(Xs, y)
    s = summarize(run_loso(Xs, y, groups), name)
    return dict(feature_set=name, n_feat=Xs.shape[1], mixed=f"{100*mm:.1f}+-{100*ms:.1f}",
                loso_subject_acc=f"{100*s['subject_acc']:.1f}",
                loso_ci95=f"[{100*s['ci95_lo']:.0f},{100*s['ci95_hi']:.0f}]",
                loso_bal_acc=f"{100*s['balanced_acc']:.1f}",
                loso_mean_seg_acc=f"{100*s['mean_segment_acc']:.1f}")


def main():
    X, y, groups = load_xy()
    res = {k: run_loso(X, y, groups, k) for k in ("xgb", "svm")}
    for k, d in res.items():
        d.to_csv(OUT / f"loso_{k}_per_subject.csv", index=False)
    tab = pd.DataFrame([summarize(d, k) for k, d in res.items()])
    print(tab.round(3).to_string(index=False))
    chance = np.bincount(y[::SEG]).max() / N_SUB
    print(f"\nMajority-class chance (subject level): {chance:.3f}")
    for k, d in res.items():
        hits = int((d.vote == d.true).sum())
        p = binomtest(hits, N_SUB, chance, alternative="greater").pvalue
        print(f"\n[{k}] {hits}/{N_SUB} subjects correct; one-sided binomial p vs chance = {p:.4f}")
        print(confusion_matrix(d.true, d.vote))
    a, b = res["xgb"], res["svm"]
    try:
        print("\nWilcoxon (per-subject segment accuracy, XGB vs SVM on the same folds):",
              wilcoxon(a.seg_acc, b.seg_acc))
    except ValueError as e:
        print("\nWilcoxon not computable:", e)
    ca, cb = (a.vote == a.true).to_numpy(), (b.vote == b.true).to_numpy()
    table = [[(ca & cb).sum(), (ca & ~cb).sum()],
             [(~ca & cb).sum(), (~ca & ~cb).sum()]]
    print("McNemar exact p (subject-level right/wrong):", mcnemar(table, exact=True).pvalue)
    mm, ms = eval_mixed(X, y)
    print(f"\nSubject-mixed 80/20 (XGB, 5 seeds): {100*mm:.1f} +- {100*ms:.1f}")


if __name__ == "__main__":
    main()