import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from common import load_xy, N_EST, SEED

# 1) train on ALL of STEW, same settings as train_xgb.py
X, y, groups = load_xy()
keep = np.load('models/keep_idx.npy')
assert np.array_equal(keep, np.flatnonzero(X.std(axis=0) > 1e-6)), 'keep_idx does not match current features'
print('STEW label values (index 0,1,2):', np.unique(np.load('data/y_labels.npy')))
print('STEW class counts:', np.bincount(y))
m = XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1,
                  eval_metric='mlogloss', random_state=SEED, n_jobs=-1).fit(X[:, keep], y)
joblib.dump(m, 'models/xgb_stew_full.pkl')

# 2) predict every Peitek window
Z = np.load('features_p01.npz')
P = m.predict_proba(Z['X'][:, keep])
pred = P.argmax(1)
print('\nPeitek predicted class counts (low, med, high):', np.bincount(pred, minlength=3), 'of', len(pred))

# 3) per-snippet load score = P(med)*1 + P(high)*2
df = pd.DataFrame({'snippet': Z['snippet'], 'p_high': P[:, 2], 'score': P[:, 1] + 2*P[:, 2]})
g = df.groupby('snippet', sort=False).agg(n_win=('score', 'size'), score=('score', 'mean'), p_high=('p_high', 'mean')).reset_index()
t = pd.read_csv('snippet_eeg_times.csv')[['order', 'snippet', 'read_dur_s']]
g = t.merge(g, on='snippet', how='left')
g.to_csv('predictions_p01.csv', index=False)
print('\n', g.round(3).to_string(index=False))
print('\nscore across snippets: min %.2f  median %.2f  max %.2f  std %.3f' % (g.score.min(), g.score.median(), g.score.max(), g.score.std()))
