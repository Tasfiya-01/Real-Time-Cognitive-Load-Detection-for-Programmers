"""SHAP for the programmer (reading-vs-rest) model, explained on held-out P05 windows.
Run from the project root:  python shap_programmer.py
Outputs: results/shap_programmer.png, results/shap_programmer.csv
"""
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
FT = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Theta/Alpha', 'Beta/Alpha', 'Theta/Beta',
      'Mean', 'Std', 'Var', 'Skew', 'Kurt', 'RMS', 'SampEn', 'PermEn', 'SpecEn', 'SVDEn']
ALL = [f"{c}_{f}" for c in CH for f in FT]

# Feature names of the 196 columns the model was trained on
keep = np.load('models/keep_idx.npy')
names = [ALL[i] for i in keep]

model = joblib.load('models/prog_model.pkl')
X = np.load('models/playback_p05.npz')['X']          # P05 was not in the training set
assert X.shape[1] == len(names), f"feature mismatch: X has {X.shape[1]}, names {len(names)}"
print('P05 windows explained:', X.shape)

# SHAP values for the "reading" (engaged) class
sv = shap.TreeExplainer(model).shap_values(X)
if isinstance(sv, list):                              # older shap: one array per class
    sv = sv[1]
elif sv.ndim == 3:                                    # newer shap: (n, features, classes)
    sv = sv[:, :, 1]
imp = np.abs(sv).mean(axis=0)

df = pd.DataFrame({'feature': names, 'mean_abs_shap': imp})
df['channel'] = df.feature.str.split('_').str[0]
df['type'] = df.feature.str.split('_').str[1]
df = df.sort_values('mean_abs_shap', ascending=False)
df.to_csv('results/shap_programmer.csv', index=False)

top = df.head(20)
ch = df.groupby('channel').mean_abs_shap.sum().reindex(CH)
ty = df.groupby('type').mean_abs_shap.sum().sort_values(ascending=False)

fig, ax = plt.subplots(1, 3, figsize=(20, 7))
ax[0].barh(top.feature[::-1], top.mean_abs_shap[::-1], color='steelblue')
ax[0].set_title('Top 20 features (programmer model, held-out P05)')
ax[0].set_xlabel('Mean |SHAP value|')
ax[1].bar(ch.index, ch.values, color=['red' if v == ch.max() else 'steelblue' for v in ch.values])
ax[1].set_title('Channel importance (sum of mean |SHAP|)')
ax[1].tick_params(axis='x', rotation=45)
ax[2].bar(ty.index, ty.values, color='steelblue')
ax[2].set_title('Feature-type importance (sum of mean |SHAP|)')
ax[2].tick_params(axis='x', rotation=45)
for a in ax:
    a.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/shap_programmer.png', dpi=150, bbox_inches='tight')

pd.set_option('display.width', 120)
print('\nTop 10 features:')
print(df.head(10)[['feature', 'mean_abs_shap']].round(4).to_string(index=False))
print('\nChannels:')
print(ch.sort_values(ascending=False).round(4).to_string())
print('\nFeature types:')
print(ty.round(4).to_string())
print('\nSaved results/shap_programmer.png and results/shap_programmer.csv')