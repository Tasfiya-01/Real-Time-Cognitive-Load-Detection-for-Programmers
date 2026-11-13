"""SHAP explanation of the difficulty-level classifier (E6, confusing vs clean) + per-person calibration test.

Uses the cached features from aoc_level_ml.py: results/aoc_level_features.csv
Same preprocessing as E6: z-score each feature within participant, regress out log reading time within participant.
SHAP values are computed on each held-out participant (LOSO), so they explain predictions on unseen people.

Usage:   python aoc_level_shap.py
Outputs: results/aoc_shap_importance.csv, results/aoc_shap_summary.txt, results/fig_aoc_shap.png
"""
from pathlib import Path
import numpy as np, pandas as pd, shap
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results")
CH = ["Fp1", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "Fp2"]
F = pd.read_csv(OUT / "aoc_level_features.csv")
COLS = [c for c in F.columns if c.split("_")[0] in CH]

# --- E6 preprocessing ---
pids = F.pid.astype(str).values
y = (F.condition == "confusing").astype(int).values
Z = F.groupby("pid")[COLS].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
T = F.groupby("pid")["read_dur"].transform(lambda s: (np.log(s) - np.log(s).mean()) / np.log(s).std(ddof=1))
X = Z.values.copy()
for pid in np.unique(pids):
    i = pids == pid
    A = np.c_[np.ones(i.sum()), T[i].values]
    coef, *_ = np.linalg.lstsq(A, X[i], rcond=None)
    X[i] = X[i] - A @ coef

def model(seed=0):
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, random_state=seed, eval_metric="logloss", n_jobs=4)

def weights(yy):
    pos = yy.mean(); return np.where(yy == 1, 0.5 / pos, 0.5 / (1 - pos))

lines = []
def say(s=""): print(s); lines.append(s)

# --- 1. SHAP over LOSO folds ---
S = np.zeros_like(X); aucs = []
for pid in np.unique(pids):
    tr, te = pids != pid, pids == pid
    m = model(); m.fit(X[tr], y[tr], sample_weight=weights(y[tr]))
    S[te] = shap.TreeExplainer(m).shap_values(X[te])
    aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
aucs = np.array(aucs)
say(f"LOSO AUC (confusing vs clean): {aucs.mean():.3f}, {(aucs > 0.5).sum()}/{len(aucs)} > 0.5, p = {wilcoxon(aucs - 0.5).pvalue:.4f}")

imp = pd.Series(np.abs(S).mean(0), index=COLS)
by_ch = imp.groupby([c.split("_")[0] for c in COLS]).sum().sort_values(ascending=False)
by_ft = imp.groupby(["_".join(c.split("_")[1:]) for c in COLS]).sum().sort_values(ascending=False)
say("\nSHAP by feature type:\n" + by_ft.round(4).to_string())
say("\nSHAP by channel:\n" + by_ch.round(4).to_string())
say("\nTop 10 features (direction: + means higher value pushes towards 'confusing'):")
for c in imp.sort_values(ascending=False).index[:10]:
    j = COLS.index(c)
    say(f"  {c:16s} {imp[c]:.4f}   direction r = {np.corrcoef(X[:, j], S[:, j])[0, 1]:+.2f}")
imp.sort_values(ascending=False).to_csv(OUT / "aoc_shap_importance.csv", header=["mean_abs_shap"])

# --- 2. Per-person calibration: add k own trials per class (weighted x10), test on the rest ---
say("\nPer-person calibration (5 repetitions):")
rng = np.random.default_rng(0)
for k in [0, 5, 10, 15]:
    reps = []
    for rep in range(1 if k == 0 else 5):
        a = []
        for pid in np.unique(pids):
            own = np.where(pids == pid)[0]
            cal = np.concatenate([rng.choice(own[y[own] == c], k, replace=False) for c in (0, 1)]).astype(int) if k else np.array([], int)
            test = np.setdiff1d(own, cal)
            tr = np.r_[np.where(pids != pid)[0], cal]
            w = weights(y[tr]); w[len(tr) - len(cal):] *= 10
            m = model(rep); m.fit(X[tr], y[tr], sample_weight=w)
            a.append(roc_auc_score(y[test], m.predict_proba(X[test])[:, 1]))
        reps.append(np.mean(a))
    say(f"  k = {k:2d} own trials per class: mean AUC {np.mean(reps):.3f}")

# --- 3. Within-person only (5-fold CV inside each participant) ---
wx, wl = [], []
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]; a, b = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx]):
        m = model(); m.fit(X[idx[tr]], y[idx[tr]])
        a.append(roc_auc_score(y[idx[te]], m.predict_proba(X[idx[te]])[:, 1]))
        l = LogisticRegression(C=0.01, max_iter=2000).fit(X[idx[tr]], y[idx[tr]])
        b.append(roc_auc_score(y[idx[te]], l.predict_proba(X[idx[te]])[:, 1]))
    wx.append(np.mean(a)); wl.append(np.mean(b))
wx, wl = np.array(wx), np.array(wl)
say(f"\nWithin-person only: XGBoost AUC {wx.mean():.3f} (p = {wilcoxon(wx - 0.5).pvalue:.3f}), "
    f"logistic regression AUC {wl.mean():.3f} (p = {wilcoxon(wl - 0.5).pvalue:.3f})")
(OUT / "aoc_shap_summary.txt").write_text("\n".join(lines), encoding="utf-8")

# --- Figure ---
fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
top = imp.sort_values().tail(10)
for a, s, t in [(ax[0], by_ft.sort_values(), "By feature type"), (ax[1], by_ch.sort_values(), "By channel"), (ax[2], top, "Top 10 features")]:
    a.barh(s.index, s.values, color=["#c27c2c" if "gamma" in i else "#2b5d8a" for i in s.index])
    a.set_title(t, fontsize=10); a.tick_params(labelsize=8)
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
ax[1].set_xlabel("Summed mean |SHAP|", fontsize=9)
plt.suptitle("What the difficulty-level classifier uses (confusing vs clean, all LOSO folds)", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "fig_aoc_shap.png", dpi=200)
print("\nSaved results/aoc_shap_summary.txt, results/aoc_shap_importance.csv, results/fig_aoc_shap.png")