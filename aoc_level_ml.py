"""E6: Can a classifier detect the LEVEL of difficulty from EEG in unseen programmers?
Atoms of Confusion data, leave-one-subject-out (24 folds).

Targets (all within the same people):
  confusing vs clean code | snippets rated difficult vs easy | wrong vs correct answers
Features per trial: log relative band power (delta..gamma) and 3 band ratios on 14 channels
(Emotiv layout, Fp1/Fp2 in place of AF3/AF4), averaged over 3-s windows (step 1.5 s), so trial length
does not change the feature scale. Each feature is z-scored within each participant (removes person identity).
Comparison: the same LOSO classifier using reading time only (behaviour).

Usage: python aoc_level_ml.py D:\\aoc_data\\raw <path to Data_Table1_behavioral_data.csv>
Outputs: results/aoc_level_features.csv, results/aoc_level_summary.txt, results/fig_aoc_level.png
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, mne
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
ROOT, BEH = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path("results"); OUT.mkdir(exist_ok=True)
CHS = ["Fp1", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "Fp2"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
WIN, STEP = 3.0, 1.5
FEAT_CACHE = OUT / "aoc_level_features.csv"

def code(d):
    m = re.search(r"S\s*(\d+)", d); return int(m.group(1)) if m else None

def window_feats(seg, fs):
    f, p = welch(seg, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f < 40)].sum(axis=1)
    bp = {b: p[:, (f >= lo) & (f < hi)].sum(axis=1) / tot for b, (lo, hi) in BANDS.items()}
    out = [np.log10(bp[b]) for b in BANDS]
    out += [np.log10(bp["theta"] / bp["alpha"]), np.log10(bp["beta"] / bp["alpha"]), np.log10(bp["theta"] / bp["beta"])]
    return np.concatenate(out)                      # 8 x 14 = 112 features

NAMES = [f"{c}_{n}" for n in list(BANDS) + ["theta/alpha", "beta/alpha", "theta/beta"] for c in CHS]

def trial_feats(data, fs, t0, t1):
    ws, t = [], t0
    while t + WIN <= t1 + 1e-9:
        ws.append(window_feats(data[:, int(t * fs):int((t + WIN) * fs)], fs)); t += STEP
    return np.mean(ws, axis=0) if ws else None

# ---------------- 1. features (cached) ----------------
if FEAT_CACHE.exists():
    F = pd.read_csv(FEAT_CACHE); print("Loaded cached features:", F.shape)
else:
    rows = []
    for f in sorted(ROOT.rglob("*.vhdr")):
        raw = mne.io.read_raw_brainvision(f, preload=False)
        if any(c not in raw.ch_names for c in CHS):
            print("skip", f.name, [c for c in CHS if c not in raw.ch_names]); continue
        raw.pick(CHS).load_data(); raw.notch_filter([50, 100]); raw.filter(1, 40)
        fs, data = raw.info["sfreq"], raw.get_data()
        ann = [(o, code(d)) for o, d in zip(raw.annotations.onset, raw.annotations.description) if code(d) is not None]
        pid, n = f.parent.name, 0
        for i, (on, c) in enumerate(ann):
            if c not in (11, 12): continue
            end = next((o for o, cc in ann[i + 1:] if cc == 4), None)
            if end is None or end - on < WIN: continue
            x = trial_feats(data, fs, on, end)
            if x is None: continue
            n += 1
            rows.append(dict(pid=pid, trial=n, condition="confusing" if c == 12 else "clean", read_dur=end - on,
                             **dict(zip(NAMES, x))))
        print(f"{f.name}: pid={pid}, trials={n}")
    F = pd.DataFrame(rows); F.to_csv(FEAT_CACHE, index=False)

# ---------------- 2. merge behaviour (rating, correctness) ----------------
b = pd.read_csv(BEH, sep=";")
b["pid"] = "E" + b.Participant.str[1:]
b["cond"] = b.SnippetCondition.map({"ambiguous": "confusing", "unambiguous": "clean"})
rows = []
for pid, g in F.groupby("pid"):
    bb = b[b.pid == pid].sort_values("TrialNo").reset_index(drop=True); j = 0
    for _, r in g.sort_values("trial").iterrows():
        k = j
        while k < len(bb) and not (bb.cond[k] == r.condition and abs(bb.ComprehensionTime[k] - r.read_dur) < 0.5): k += 1
        if k < len(bb):
            rows.append({**r.to_dict(), "difficult": 1 - int(bb.SubjectiveDifficultyRating[k]), "wrong": 1 - int(bb.AnswerCorrectness[k])}); j = k + 1
D = pd.DataFrame(rows)
print("Matched trials:", len(D), "of", len(F))
D["confusing"] = (D.condition == "confusing").astype(int)

# within-person z-score of every feature (label-free)
X = D[NAMES].copy()
X = D.groupby("pid")[NAMES].transform(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
T = D.groupby("pid")["read_dur"].transform(lambda s: (s - s.mean()) / s.std(ddof=1)).to_frame()

# ---------------- 3. LOSO ----------------
def loso(Xm, y, model):
    aucs = {}
    for pid in D.pid.unique():
        tr, te = D.pid != pid, D.pid == pid
        if y[te].nunique() < 2 or y[te].sum() < 2: continue
        pos = y[tr].mean()
        m = model(); m.fit(Xm[tr], y[tr], sample_weight=np.where(y[tr] == 1, 0.5 / pos, 0.5 / (1 - pos)))
        aucs[pid] = roc_auc_score(y[te], m.predict_proba(Xm[te])[:, 1])
    return pd.Series(aucs)

xgb = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.5, random_state=42, eval_metric="logloss")
lines, res = [], {}
def say(s=""): print(s); lines.append(s)
for tgt, label in [("confusing", "Confusing vs clean"), ("difficult", "Rated difficult vs easy"), ("wrong", "Wrong vs correct")]:
    y = D[tgt]
    a_eeg = loso(X, y, xgb); a_time = loso(T, y, xgb)
    res[label] = (a_eeg, a_time)
    say(f"\n{label}  (positive share {y.mean():.2f})")
    say(f"  EEG features : mean AUC {a_eeg.mean():.3f} +- {a_eeg.std():.3f}, {(a_eeg > 0.5).sum()}/{len(a_eeg)} above 0.5, Wilcoxon p = {wilcoxon(a_eeg - 0.5).pvalue:.4f}")
    say(f"  Reading time : mean AUC {a_time.mean():.3f} +- {a_time.std():.3f}, {(a_time > 0.5).sum()}/{len(a_time)} above 0.5, Wilcoxon p = {wilcoxon(a_time - 0.5).pvalue:.4f}")

# reading vs fixation is not modelled here (no baseline features); see aoc_index.py
(OUT / "aoc_level_summary.txt").write_text("\n".join(lines), encoding="utf-8")

fig, ax = plt.subplots(figsize=(8, 4.2))
labels = list(res)
for k, (lab, (e, t)) in enumerate(res.items()):
    ax.scatter(np.full(len(e), k - 0.15) + np.random.uniform(-0.05, 0.05, len(e)), e, s=14, color="#2b5d8a")
    ax.scatter(np.full(len(t), k + 0.15) + np.random.uniform(-0.05, 0.05, len(t)), t, s=14, color="#c27c2c")
    ax.plot([k - 0.28, k - 0.02], [e.mean()] * 2, color="#2b5d8a", lw=2.5)
    ax.plot([k + 0.02, k + 0.28], [t.mean()] * 2, color="#c27c2c", lw=2.5)
ax.axhline(0.5, color="black", lw=0.8, ls="--")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("AUC on held-out participant"); ax.set_ylim(0.2, 1.0)
ax.scatter([], [], color="#2b5d8a", label="EEG features"); ax.scatter([], [], color="#c27c2c", label="Reading time only")
ax.legend(loc="upper right", fontsize=9); ax.set_title("Difficulty level from EEG, leave-one-subject-out (24 programmers)")
plt.tight_layout(); plt.savefig(OUT / "fig_aoc_level.png", dpi=180)
print("\nSaved results/aoc_level_summary.txt and results/fig_aoc_level.png")