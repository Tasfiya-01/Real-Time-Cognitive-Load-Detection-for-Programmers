"""E7: Difficulty level from time-locked EEG (Atoms of Confusion), leave-one-subject-out.

Uses the authors' preprocessed single-trial epochs (ICA-cleaned, re-referenced):
  FRP: locked to the first fixation on the confusing / clean region (-300..1000 ms) -> needs eye tracking
  ERP: locked to snippet onset (-400..2000 ms)                                        -> needs no eye tracker
File names contain 'obf' (confusing) or 'clean'.

Features: mean amplitude in 100-ms bins after the event, on every channel, baseline -100..0 ms.
Each feature z-scored within participant (label-free). Classifiers: shrinkage LDA (standard for ERP decoding)
and logistic regression. Also the authors' univariate test: frontal mean amplitude 400-700 ms.

Usage: python aoc_frp_decode.py D:\\aoc_results\\results
Outputs: results/aoc_frp_summary.txt, results/aoc_frp_auc.csv, results/fig_aoc_frp.png
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, mne
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
ROOT = Path(sys.argv[1])
OUT = Path("results"); OUT.mkdir(exist_ok=True)
FRONTAL = ["Fz", "F3", "F4", "FC3", "FCz", "FC4"]

def load(kind, tmax):
    rows, X, chans = [], [], None
    folder = ROOT / f"EEG_{kind}" / "epochs"
    for pdir in sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "total"):
        n = 0
        for f in sorted(pdir.glob("*_epo.fif")):
            m = re.search(r"(clean|obf)", f.name)
            if not m: continue
            ep = mne.read_epochs(f, preload=True)
            ep = ep.pick("eeg")
            if chans is None: chans = ep.ch_names
            if ep.ch_names != chans: ep = ep.reorder_channels(chans)
            ep.apply_baseline((-0.1, 0.0))
            d = ep.get_data().mean(axis=0)                     # (channels, times); one trial per file
            t = ep.times
            bins = [d[:, (t >= a) & (t < a + 0.1)].mean(axis=1) for a in np.arange(0.0, tmax, 0.1)]
            fr = [chans.index(c) for c in FRONTAL if c in chans]
            late = d[fr][:, (t >= 0.4) & (t < 0.7)].mean()
            X.append(np.concatenate(bins)); n += 1
            rows.append(dict(pid=pdir.name, file=f.name, confusing=int(m.group(1) == "obf"), frontal_400_700=late))
        print(f"{kind} {pdir.name}: {n} epochs")
    D = pd.DataFrame(rows); X = np.array(X)
    names = [f"{c}_{int(a*1000)}ms" for a in np.arange(0.0, tmax, 0.1) for c in chans]
    return D, pd.DataFrame(X, columns=names)

def zscore_within(D, X):
    return X.groupby(D.pid.values).transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))

def loso(D, X, mk):
    out = {}
    for pid in D.pid.unique():
        tr, te = (D.pid != pid).values, (D.pid == pid).values
        y = D.confusing.values
        if len(set(y[te])) < 2: continue
        m = mk(); m.fit(X[tr], y[tr])
        out[pid] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    return pd.Series(out)

lines = []
def say(s=""): print(s); lines.append(s)
allres = {}
for kind, tmax, label in [("FRP", 1.0, "Fixation-locked (needs eye tracking)"), ("ERP", 2.0, "Snippet-onset-locked (no eye tracker)")]:
    D, X = load(kind, tmax)
    Z = zscore_within(D, X).values
    say(f"\n=== {kind}: {label} ===")
    say(f"Participants {D.pid.nunique()}, epochs {len(D)} (confusing {D.confusing.sum()}, clean {(1-D.confusing).sum()})")
    # univariate: authors' late frontal positivity
    diff = D.groupby(["pid", "confusing"]).frontal_400_700.mean().unstack()
    dd = (diff[1] - diff[0]).dropna() * 1e6
    say(f"Frontal 400-700 ms, confusing - clean: {dd.mean():+.2f} uV, {(dd > 0).sum()}/{len(dd)} positive, Wilcoxon p = {wilcoxon(dd).pvalue:.4f}")
    for name, mk in [("Shrinkage LDA", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                     ("Logistic regression", lambda: LogisticRegression(C=0.01, max_iter=3000))]:
        a = loso(D, Z, mk); allres[(kind, name)] = a
        say(f"{name:20s}: AUC {a.mean():.3f} +- {a.std():.3f}, {(a > 0.5).sum()}/{len(a)} > 0.5, Wilcoxon p = {wilcoxon(a - 0.5).pvalue:.5f}")
(OUT / "aoc_frp_summary.txt").write_text("\n".join(lines), encoding="utf-8")
pd.DataFrame({f"{k}_{n}": v for (k, n), v in allres.items()}).to_csv(OUT / "aoc_frp_auc.csv")

fig, ax = plt.subplots(figsize=(7, 4.2)); rng = np.random.default_rng(0)
keys = list(allres)
for i, k in enumerate(keys):
    a = allres[k]
    ax.scatter(i + rng.uniform(-0.08, 0.08, len(a)), a, s=14, color="#2b5d8a" if k[0] == "FRP" else "#c27c2c")
    ax.plot([i - 0.2, i + 0.2], [a.mean()] * 2, color="black", lw=2)
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"{k}\n{n}" for k, n in keys], fontsize=9)
ax.set_ylabel("AUC, held-out participant"); ax.set_title("Confusing vs clean from time-locked EEG (LOSO)")
plt.tight_layout(); plt.savefig(OUT / "fig_aoc_frp.png", dpi=180)
print("\nSaved results/aoc_frp_summary.txt, results/aoc_frp_auc.csv, results/fig_aoc_frp.png")