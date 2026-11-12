import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, mne

fif = r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif'
times = pd.read_csv('snippet_eeg_times.csv')
STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
raw.pick(STEW).load_data()
raw.filter(1, 45, verbose=False)
raw.resample(128)
fs = 128
data = raw.get_data()

X, names = [], []
for _, r in times.iterrows():
    a, b = int(r.read_start_s * fs), int(r.read_end_s * fs)
    X.append(data[:, a:b]); names.append(r.snippet)

np.savez('segments_p01.npz', **{n: x for n, x in zip(names, X)})
print('saved', len(X), 'segments')
rs = [1.4826*np.median(np.abs(x-np.median(x,axis=1,keepdims=True))) for x in X]
mx = [np.abs(x).max() for x in X]
print('IsPrime shape:', X[0].shape)
print('robust std per segment: min %.1f median %.1f max %.1f' % (min(rs), np.median(rs), max(rs)))
print('segments with max abs > 500:', sum(m > 500 for m in mx), 'of', len(X))
