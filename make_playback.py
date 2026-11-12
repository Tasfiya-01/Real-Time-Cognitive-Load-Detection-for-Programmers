"""Train the programmer (reading-vs-rest) model on 9 participants and save the 10th for EEG playback.
Run from the project root:  python make_playback.py
"""
import sys, glob, re
sys.path.insert(0, 'src')
import numpy as np, joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from common import N_EST, SEED

PLAY = '05'                                   # this participant is NOT used for training
keep = np.load('models/keep_idx.npy')
data = {re.search(r'rr_p(\d+)', f).group(1): np.load(f) for f in sorted(glob.glob('results/rr_p*.npz'))}

Xtr = np.vstack([data[p]['X'][:, keep] for p in data if p != PLAY])
ytr = np.concatenate([data[p]['y'] for p in data if p != PLAY])
w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
model = XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                      random_state=SEED, n_jobs=-1).fit(Xtr, ytr, sample_weight=w)
joblib.dump(model, 'models/prog_model.pkl')

# playback data: P05's windows in recording order (each snippet's reading, then its rest)
X, y, s = data[PLAY]['X'][:, keep], data[PLAY]['y'], data[PLAY]['snippet']
np.savez('models/playback_p05.npz', X=X, y=y, snippet=s)

p = model.predict_proba(X)[:, 1]
print(f'trained on {len(set(data) - {PLAY})} participants, {len(ytr)} windows')
print(f'playback participant P{PLAY}: {len(y)} windows = {len(y) * 5 / 60:.1f} min, AUC {roc_auc_score(y, p):.3f}')
print('saved models/prog_model.pkl and models/playback_p05.npz')