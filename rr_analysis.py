"""Reading vs rest: (A) STEW model, (B) LOSO model trained on programmers, (C) channel check for eye movements.
Run from the project root:  python rr_analysis.py
"""
import sys, glob, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import wilcoxon
from common import N_EST, SEED

keep = np.load('models/keep_idx.npy')
stew = joblib.load('models/xgb_stew_full.pkl')

# load cached reading/rest features
data = {}
for f in sorted(glob.glob('results/rr_p*.npz')):
    pid = re.search(r'rr_p(\d+)', f).group(1)
    z = np.load(f)
    data[pid] = (z['X'], z['y'])
pids = list(data)
print('participants:', ', '.join(pids))

# (A) STEW model: does it give higher load during reading than rest?
rows = []
for pid, (X, y) in data.items():
    P = stew.predict_proba(X[:, keep])
    s = P[:, 1] + 2*P[:, 2]
    rows.append(dict(pid=pid, n_read=int((y == 1).sum()), n_rest=int((y == 0).sum()),
                     stew_read=s[y == 1].mean(), stew_rest=s[y == 0].mean(), stew_auc=roc_auc_score(y, s)))

# (B) + (C) leave-one-subject-out on the programmer data, three channel sets
# STEW channel index: AF3 0, F7 1, F3 2, FC5 3, T7 4, P7 5, O1 6, O2 7, P8 8, T8 9, FC6 10, F4 11, F8 12, AF4 13
SETS = {'all': list(range(14)), 'posterior': [5, 6, 7, 8], 'frontal': [0, 1, 2, 11, 12, 13]}
for name, chs in SETS.items():
    cols = np.array([c for c in keep if c // 18 in chs])
    for r in rows:
        test = r['pid']
        Xtr = np.vstack([data[p][0][:, cols] for p in pids if p != test])
        ytr = np.concatenate([data[p][1] for p in pids if p != test])
        w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))     # balance reading/rest
        m = XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                          random_state=SEED, n_jobs=-1).fit(Xtr, ytr, sample_weight=w)
        Xte, yte = data[test][0][:, cols], data[test][1]
        p = m.predict_proba(Xte)[:, 1]
        r[f'{name}_auc'] = roc_auc_score(yte, p)
        r[f'{name}_balacc'] = balanced_accuracy_score(yte, p > 0.5)
    print(f'LOSO done: {name} ({len(cols)} features)', flush=True)

df = pd.DataFrame(rows)
df.to_csv('results/rr_summary.csv', index=False)
print('\nper participant:')
print(df.round(3).to_string(index=False))

print('\ngroup summary (AUC: 0.5 = chance, 1.0 = perfect):')
for col, label in [('stew_auc', '(A) STEW model, no retraining'),
                   ('all_auc', '(B) LOSO, all 14 channels'),
                   ('posterior_auc', '(C) LOSO, posterior only (P7 O1 O2 P8)'),
                   ('frontal_auc', '(C) LOSO, frontal only (AF3 F7 F3 F4 F8 AF4)')]:
    v = df[col].values
    print(f'  {label:45s} mean AUC {v.mean():.3f} (sd {v.std():.3f}) | above 0.5: {int((v > 0.5).sum())}/{len(v)} | Wilcoxon vs 0.5 p = {wilcoxon(v - 0.5).pvalue:.4f}')
for col in ['all_balacc', 'posterior_balacc', 'frontal_balacc']:
    print(f'  balanced accuracy {col.replace("_balacc", ""):10s} mean {df[col].mean():.3f}')
print(f'  (A) STEW mean score reading {df.stew_read.mean():.3f} vs rest {df.stew_rest.mean():.3f}, paired Wilcoxon p = {wilcoxon(df.stew_read, df.stew_rest).pvalue:.4f}')