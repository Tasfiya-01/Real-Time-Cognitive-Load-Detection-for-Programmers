import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.io as sio, pandas as pd, mne
from scipy.signal import welch

STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
BANDS = [(0.5,4),(4,8),(8,13),(13,30),(30,43),(43,48),(48,52),(52,64)]

def rel(f, P, lo, hi):                       # share of power in [lo,hi) out of 0.5-64 Hz
    tot = P[(f >= 0.5) & (f < 64)].sum()
    return P[(f >= lo) & (f < hi)].sum() / tot

# STEW: raw, 128 Hz, average PSD over channels and recordings
d = sio.loadmat(r'data\dataset.mat')['dataset'].astype(float)          # (14, 19200, 45)
fs_, Ps = welch(d - d.mean(axis=1, keepdims=True), fs=128, nperseg=512, axis=1)
Ps = Ps.mean(axis=(0, 2))

# Peitek: raw 500 Hz, only the code-reading segments
raw = mne.io.read_raw_fif(r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif', preload=False, verbose=False)
raw.pick(STEW).load_data(); raw.reorder_channels(STEW)
FS = raw.info['sfreq']; x = raw.get_data()
times = pd.read_csv('snippet_eeg_times.csv')
Pp, w = 0, 0
for _, r in times.iterrows():
    seg = x[:, int(r.read_start_s*FS):int(r.read_end_s*FS)]
    if seg.shape[1] < 4*FS: continue
    fp, P = welch(seg - seg.mean(1, keepdims=True), fs=FS, nperseg=int(4*FS), axis=1)
    Pp = Pp + P.mean(0)*seg.shape[1]; w += seg.shape[1]
Pp = Pp / w

print('Peitek sampling rate:', FS)
print('\nshare of 0.5-64 Hz power   STEW    Peitek')
for lo, hi in BANDS:
    print(f'  {lo:>4}-{hi:<4} Hz          {rel(fs_,Ps,lo,hi):6.3f}   {rel(fp,Pp,lo,hi):6.3f}')

def peak_ratio(f, P, f0):                    # power at f0 vs median of 40-47 Hz (1 = no peak)
    return P[np.argmin(abs(f-f0))] / np.median(P[(f >= 40) & (f < 47)])
print('\n50 Hz peak ratio (1 = no peak):  STEW %.1f   Peitek %.1f' % (peak_ratio(fs_,Ps,50), peak_ratio(fp,Pp,50)))
print('60 Hz peak ratio (1 = no peak):  STEW %.1f   Peitek %.1f' % (peak_ratio(fs_,Ps,60), peak_ratio(fp,Pp,60)))
hi_f = (fp >= 20) & (fp < FS/2)
print('Peitek top 5 peak frequencies above 20 Hz:', np.round(fp[hi_f][np.argsort(-Pp[hi_f])[:5]], 2))
