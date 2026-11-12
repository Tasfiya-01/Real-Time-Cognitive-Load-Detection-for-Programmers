import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd, mne
from scipy import signal as sig
from old_features_v3 import extract_features_v3

fif = r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif'
times = pd.read_csv('snippet_eeg_times.csv')
STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FS, WIN = 128, 640
FN = ['delta','theta','alpha','beta','gamma','t/a','b/a','t/b','mean','std','var','skew','kurt','rms','sampEn','permEn','specEn','svdEn']

raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
raw.pick(STEW).load_data()
raw.reorder_channels(STEW); print('channel order:', raw.ch_names)   # NEW: force STEW channel order
raw.notch_filter([50, 100, 150, 200], verbose=False)   # remove 50 Hz mains noise + harmonics (STEW has none)
raw.filter(None, 43, verbose=False)                    # low-pass 43 Hz to match Emotiv/STEW bandwidth
raw.resample(FS)
data = raw.get_data().T

b, a = sig.butter(4, [0.5/64, 50.0/64], btype='band')
F, owner = [], []
for _, r in times.iterrows():
    s, e = int(r.read_start_s*FS), int(r.read_end_s*FS)
    seg = data[s:e]
    for k in range(len(seg)//WIN):
        w = seg[k*WIN:(k+1)*WIN]
        w = sig.filtfilt(b, a, w, axis=0)
        w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        F.append(extract_features_v3(w)); owner.append(r.snippet)
F = np.array(F)
np.savez('features_p01.npz', X=F, snippet=np.array(owner))
print('features:', F.shape, 'NaN:', int(np.isnan(F).sum()))

S = np.load('data/X_features_v3_fixed.npy')
lo, hi = np.percentile(S, 1, axis=0), np.percentile(S, 99, axis=0)
out = ((F < lo) | (F > hi)).mean(0)
print('mean fraction outside STEW 1-99 pct:', round(float(out.mean()), 3), '(about 0.02 would be normal)')
for i in np.argsort(-out)[:10]:
    print(' ', STEW[i//18], FN[i%18], round(float(out[i]), 2))