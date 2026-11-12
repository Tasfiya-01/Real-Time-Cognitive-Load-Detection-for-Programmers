"""Clean XGBoost training: no scaler, constant columns dropped, two evaluation protocols.
Run from the project root:  python src\\train_xgb.py
Uses the same load_xy() / FEATURE_FILE as loso_eval.py, so all thesis numbers share one source.
"""
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from common import *

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

print("Loading features...")
X, y, groups = load_xy()
classes = [str(c) for c in np.unique(np.load(DATA / "y_labels.npy"))]
print("Classes in index order (0,1,2,...):", classes)

# Drop (near-)constant columns (the 56 Mean/Std/Var/RMS columns). Label-free.
keep = np.flatnonzero(X.std(axis=0) > 1e-6)
X = X[:, keep]
print(f"Kept {len(keep)} features, X shape: {X.shape}")
np.save(MODELS / "keep_idx.npy", keep)   # the SHAP script needs this to map names back


def make_model():
    return XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1,
                         eval_metric="mlogloss", random_state=SEED, n_jobs=-1)


def run(title, tr, te):
    m = make_model()
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    acc = accuracy_score(y[te], p)
    print("\n" + "=" * 60)
    print(f"{title}: {acc*100:.2f}%  (train {len(tr)}, test {len(te)})")
    print("=" * 60)
    print(classification_report(y[te], p, target_names=classes, zero_division=0))
    return m, p, acc


# 1) Subject-mixed stratified 80/20: the old headline protocol (segments of the same
#    subject appear in train AND test, so it is leaky).
tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y))
m_mixed, p_mixed, acc_mixed = run("Subject-MIXED 80/20 (leaky)", tr, te)

# 2) Subject-held-out 80/20: whole subjects are test-only. Single split of ~9 subjects,
#    so it is noisy; the LOSO numbers from loso_eval.py are the reliable ones.
tr2, te2 = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y, groups))
m_subj, p_subj, acc_subj = run("Subject-HELD-OUT 80/20", tr2, te2)

print("\nSaving model...")
joblib.dump(m_mixed, MODELS / "xgb_model_clean.pkl")
print("Saved models/xgb_model_clean.pkl and models/keep_idx.npy")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (t, yt, pt, a) in zip(axes, [("Subject-mixed (leaky)", y[te], p_mixed, acc_mixed),
                                     ("Subject-held-out", y[te2], p_subj, acc_subj)]):
    sns.heatmap(confusion_matrix(yt, pt), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"{t}\nAccuracy: {a*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT / "xgb_results_clean.png", dpi=150)
print("Plot saved: results/xgb_results_clean.png")