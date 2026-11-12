import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, mne
from scipy.signal import butter, sosfiltfilt

fif = r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif'
times = pd.read_csv('snippet_eeg_times.csv')
STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FS, WIN = 128, 640

raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
raw.pick(STEW).load_data()
raw.resample(FS)
data = raw.get_data().T                     # (time, 14)
sos = butter(4, [0.5, 50], btype="band", fs=FS, output="sos")
data = sosfiltfilt(sos, data, axis=0)       # STEW-এর মতোই 0.5-50 Hz

wins, owner = [], []
for _, r in times.iterrows():
    a, b = int(r.read_start_s * FS), int(r.read_end_s * FS)
    seg = data[a:b]
    for k in range(len(seg) // WIN):        # 5 সেকেন্ডের অ-ওভারল্যাপ উইন্ডো
        w = seg[k*WIN:(k+1)*WIN]
        wins.append((w - w.mean(0)) / (w.std(0) + 1e-8))   # প্রতি উইন্ডো, প্রতি চ্যানেল z-score
        owner.append(r.snippet)

np.savez('windows_p01.npz', X=np.stack(wins), snippet=np.array(owner))
print('windows:', np.stack(wins).shape)
print('std check (should be ~1):', np.stack(wins).std())
print('windows per snippet: min', min(np.unique(owner, return_counts=True)[1]),
      'max', max(np.unique(owner, return_counts=True)[1]))
