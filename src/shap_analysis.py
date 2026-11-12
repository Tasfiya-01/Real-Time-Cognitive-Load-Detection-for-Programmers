"""SHAP analysis on the clean model: no scaler, 196 non-constant features, test rows from all subjects.
Run after train_xgb.py:  python src\\shap_analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import StratifiedShuffleSplit
from common import *

print("Loading data and model...")
X, y, groups = load_xy()
keep = np.load(ROOT / "models" / "keep_idx.npy")
model = joblib.load(ROOT / "models" / "xgb_model_clean.pkl")
X = X[:, keep]

# Feature names, same order as the extraction code (18 features per channel).
feature_types = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma',
                 'Theta/Alpha', 'Beta/Alpha', 'Theta/Beta',
                 'Mean', 'Std', 'Var', 'Skew', 'Kurt', 'RMS',
                 'SampEn', 'PermEn', 'SpecEn', 'SVDEn']
all_names = [f"{ch}_{ft}" for ch in CH for ft in feature_types]
feature_names = [all_names[i] for i in keep]
assert len(feature_names) == X.shape[1]

# Same stratified 80/20 split as train_xgb.py (it depends only on y and SEED), so these
# are the held-out rows of the saved model, spread over many subjects.
_, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED).split(X, y))
Xs = X[te]
print(f"Explaining {len(te)} held-out rows from {len(np.unique(groups[te]))} subjects")

print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xs)
if isinstance(sv, list):          # old shap: list of (n, feat) per class
    sv = np.stack(sv, axis=-1)    # -> (n, feat, classes)
sv = np.abs(sv)
shap_abs = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(axis=0)

top_idx = np.argsort(shap_abs)[::-1][:20]
top_names = [feature_names[i] for i in top_idx]
top_vals = shap_abs[top_idx]

channel_importance = {}
for name, v in zip(feature_names, shap_abs):
    ch = name.split('_')[0]
    channel_importance[ch] = channel_importance.get(ch, 0) + v

print("Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes[0].barh(range(20), top_vals, color='steelblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top_names, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_title('Top 20 Features (SHAP, clean 196-feature model)', fontsize=13)
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].grid(True, alpha=0.3)

ch_names = list(channel_importance.keys())
ch_vals = list(channel_importance.values())
colors = ['red' if v == max(ch_vals) else 'steelblue' for v in ch_vals]
axes[1].bar(ch_names, ch_vals, color=colors)
axes[1].set_title('EEG Channel Importance (SHAP)', fontsize=13)
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Total mean |SHAP|')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "shap_analysis_clean.png", dpi=150, bbox_inches='tight')
print("Saved! results/shap_analysis_clean.png")

print("\nTop 5 Most Important Features:")
for i, (name, val) in enumerate(zip(top_names[:5], top_vals[:5])):
    print(f"  {i+1}. {name}: {val:.4f}")

print("\nTop 3 Most Important Channels:")
sorted_ch = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
for i, (ch, val) in enumerate(sorted_ch[:3]):
    print(f"  {i+1}. {ch}: {val:.4f}")