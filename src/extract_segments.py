import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, mne

fif = r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif'
times = pd.read_csv('snippet_eeg_times.csv')

STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
raw.pick(STEW).load_data()
raw.resample(128)  # anti-alias filter লাগিয়ে ৫০০ -> ১২৮ Hz
fs = 128

X, names = [], []
for _, r in times.iterrows():
    a, b = int(r.read_start_s * fs), int(r.read_end_s * fs)
    X.append(raw.get_data()[:, a:b] * 1e6)   # volt -> microvolt
    names.append(r.snippet)

np.savez('segments_p01.npz', **{n: x for n, x in zip(names, X)})
print('saved', len(X), 'segments')
for n, x in zip(names[:5], X[:5]):
    print(n, x.shape, 'std(uV):', round(float(x.std()), 1))
print('overall std (uV):', round(float(np.mean([x.std() for x in X])), 1))