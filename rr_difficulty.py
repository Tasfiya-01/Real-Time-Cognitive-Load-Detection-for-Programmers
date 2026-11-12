"""Does the programmer-trained model see more 'reading-like' load on harder snippets?
Two variants: raw features, and calibrated features (z-scored against each person's own rest windows).
Run from the project root:  python rr_difficulty.py
"""
import sys, glob, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr
from common import N_EST, SEED

keep = np.load('models/keep_idx.npy')
cc = pd.read_csv('snippet_complexity.csv').set_index('snippet')

data = {}
for f in sorted(glob.glob('results/rr_p*.npz')):
    pid = re.search(r'rr_p(\d+)', f).group(1)
    z = np.load(f)
    data[pid] = (z['X'][:, keep], z['y'], z['snippet'])
pids = list(data)


def calibrate(X, y):
    """z-score every feature against this person's own rest windows (personal baseline)."""
    mu, sd = X[y == 0].mean(0), X[y == 0].std(0) + 1e-8
    return (X - mu) / sd


all_rows = []
for variant in ['raw', 'calibrated']:
    D = {p: ((calibrate(X, y) if variant == 'calibrated' else X), y, s) for p, (X, y, s) in data.items()}
    per_snip = []
    rows = []
    for test in pids:
        Xtr = np.vstack([D[p][0] for p in pids if p != test])
        ytr = np.concatenate([D[p][1] for p in pids if p != test])
        w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
        m = XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                          random_state=SEED, n_jobs=-1).fit(Xtr, ytr, sample_weight=w)
        X, y, s = D[test]
        p = m.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        # per-snippet mean "reading-likeness" during code reading only
        g = pd.DataFrame({'snippet': s[y == 1], 'p': p[y == 1]}).groupby('snippet').p.mean()
        g = g.to_frame().join(cc[['cc_max']], how='inner')
        r, pv = spearmanr(g.cc_max, g.p)
        rows.append(dict(variant=variant, pid=test, auc=auc, rho_cc_max=r, p_cc_max=pv, n_snip=len(g)))
        gz = (g.p - g.p.mean()) / g.p.std()
        per_snip.append(pd.DataFrame({'snippet': g.index, 'z': gz.values}))
    R = pd.DataFrame(rows)
    all_rows.append(R)
    pooled = pd.concat(per_snip).groupby('snippet').z.mean().to_frame().join(cc[['cc_max']])
    pr, pp = spearmanr(pooled.cc_max, pooled.z)
    print(f'\n=== {variant} features ===')
    print(R[['pid', 'n_snip', 'auc', 'rho_cc_max', 'p_cc_max']].round(3).to_string(index=False))
    print(f'reading vs rest AUC: mean {R.auc.mean():.3f} (sd {R.auc.std():.3f}), above 0.5: {int((R.auc > 0.5).sum())}/{len(R)}')
    print(f'difficulty: median rho {R.rho_cc_max.median():+.3f}, positive {int((R.rho_cc_max > 0).sum())}/{len(R)}, '
          f'Wilcoxon p = {wilcoxon(R.rho_cc_max).pvalue:.3f} | pooled rho {pr:+.3f}, p = {pp:.3f}')

pd.concat(all_rows).to_csv('results/rr_difficulty.csv', index=False)
print('\nsaved results/rr_difficulty.csv')