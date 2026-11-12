"""Run the full STEW -> Peitek transfer test for ONE participant.
Usage: python run_participant.py <path_to_fif> <participant_id>
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, mne, joblib
from scipy import signal as sig
from scipy.stats import spearmanr
from old_features_v3 import extract_features_v3

fif, pid = sys.argv[1], sys.argv[2]
STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FS, WIN = 128, 640
CC = pd.read_csv('snippet_complexity.csv')
NAMES = sorted(CC.snippet, key=str.lower)        # snippet id = case-insensitive alphabetical index

# 1) snippet times from the markers stored in info['events'] (authors' method)
raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
fs0 = raw.info['sfreq']
ev = np.array([[e['list'][2], e['list'][1], e['list'][0]] for e in raw.info['events'] if e.get('list') is not None])
ids, t = ev[:, 0], ev[:, 2] / fs0
T = pd.DataFrame([(NAMES[int(ids[i])], t[i+1], t[i+2]) for i in range(len(ids))
                  if ids[i] <= 100 and i + 2 < len(ids) and int(ids[i]) < len(NAMES)],
                 columns=['snippet', 'start', 'end'])
T['read_dur_s'] = T.end - T.start
print(f'[{pid}] sfreq {fs0}, first_samp {raw.first_samp}, snippets found: {len(T)}')

# 2) harmonise with STEW: channel order, 50 Hz notch, 43 Hz low-pass, 128 Hz
raw.pick(STEW).load_data(); raw.reorder_channels(STEW)
raw.notch_filter([50, 100, 150, 200], verbose=False)
raw.filter(None, 43, verbose=False)
raw.resample(FS)
data = raw.get_data().T

# 3) same windowing + features as STEW
b, a = sig.butter(4, [0.5/64, 50.0/64], btype='band')
F, owner = [], []
for _, r in T.iterrows():
    seg = data[int(r.start*FS):int(r.end*FS)]
    for k in range(len(seg) // WIN):
        w = sig.filtfilt(b, a, seg[k*WIN:(k+1)*WIN], axis=0)
        w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        F.append(extract_features_v3(w)); owner.append(r.snippet)
F = np.array(F)

# 4) STEW model -> per-snippet load score
m = joblib.load('models/xgb_stew_full.pkl'); keep = np.load('models/keep_idx.npy')
S = np.load('data/X_features_v3_fixed.npy')
lo, hi = np.percentile(S, 1, axis=0), np.percentile(S, 99, axis=0)
shift = float(((F < lo) | (F > hi)).mean(0)[keep].mean())
P = m.predict_proba(F[:, keep])
df = pd.DataFrame({'snippet': owner, 'score': P[:, 1] + 2*P[:, 2]})
g = df.groupby('snippet').agg(n_win=('score', 'size'), score=('score', 'mean')).reset_index()
g = T.merge(g, on='snippet', how='left').merge(CC, on='snippet', how='left').dropna(subset=['score'])
os.makedirs('results', exist_ok=True)
g.to_csv(f'results/p{pid}_snippets.csv', index=False)

# 5) correlations and one summary row per participant
row = {'pid': pid, 'n_snippets': len(g), 'n_windows': len(F), 'shift_kept': round(shift, 3),
       'pred_low_med_high': str(np.bincount(P.argmax(1), minlength=3).tolist()), 'score_std': round(g.score.std(), 3)}
for k in ['cc_max', 'cc_sum', 'nloc', 'read_dur_s']:
    r, p = spearmanr(g[k], g.score); row[f'rho_{k}'] = round(r, 3); row[f'p_{k}'] = round(p, 3)
print(pd.Series(row).to_string())
out = 'results/summary_participants.csv'
s = pd.read_csv(out, dtype={'pid': str}) if os.path.exists(out) else pd.DataFrame()
s = pd.concat([s[s.pid != pid] if len(s) else s, pd.DataFrame([row])], ignore_index=True)
s.to_csv(out, index=False); print(f'\nsaved {out} ({len(s)} participants)')
