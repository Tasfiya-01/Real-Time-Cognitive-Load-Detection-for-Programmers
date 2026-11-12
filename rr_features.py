"""Reading-vs-rest features for every participant (cached per participant).
Run from the project root:  python rr_features.py
"""
import sys, os, glob, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, mne
from scipy import signal as sig
from old_features_v3 import extract_features_v3

ROOT = r'C:\Users\HP\Downloads\Participant'
STEW = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FS, WIN = 128, 640
NAMES = sorted(pd.read_csv('snippet_complexity.csv').snippet, key=str.lower)
b, a = sig.butter(4, [0.5/64, 50.0/64], btype='band')
os.makedirs('results', exist_ok=True)


def windows(data, start, end):
    """5-s windows inside [start, end) seconds, same preprocessing as STEW."""
    seg = data[int(start*FS):int(end*FS)]
    out = []
    for k in range(len(seg) // WIN):
        w = sig.filtfilt(b, a, seg[k*WIN:(k+1)*WIN], axis=0)
        w = (w - w.mean(0)) / (w.std(0) + 1e-8)
        out.append(extract_features_v3(w))
    return out


for folder in sorted(glob.glob(os.path.join(ROOT, 'Participant*'))):
    pid = re.search(r'Participant(\d+)', folder).group(1).zfill(2)
    out = f'results/rr_p{pid}.npz'
    fifs = glob.glob(os.path.join(folder, '*.fif'))
    if pid == '09' or not fifs or os.path.exists(out):
        continue

    raw = mne.io.read_raw_fif(fifs[0], preload=False, verbose=False)
    fs0 = raw.info['sfreq']
    ev = np.array([[e['list'][2], e['list'][0]] for e in raw.info['events'] if e.get('list') is not None])
    ids, t = ev[:, 0], ev[:, 1] / fs0
    snip_idx = [i for i in range(len(ids)) if ids[i] <= 100 and i + 3 < len(ids)]

    # harmonise with STEW (same as run_participant.py)
    raw.pick(STEW).load_data(); raw.reorder_channels(STEW)
    raw.notch_filter([50, 100, 150, 200], verbose=False)
    raw.filter(None, 43, verbose=False)
    raw.resample(FS)
    data = raw.get_data().T

    X, y, snip = [], [], []
    for n, i in enumerate(snip_idx):
        name = NAMES[int(ids[i])]
        # reading: code on screen
        for f in windows(data, t[i+1], t[i+2]):
            X.append(f); y.append(1); snip.append(name)
        # rest: fixation cross, 29 s to 1 s before the next snippet marker
        if n + 1 < len(snip_idx):
            nxt = t[snip_idx[n+1]]
            r0, r1 = nxt - 29, nxt - 1
            if r0 > t[i+3]:                       # must start after the answer screen began
                for f in windows(data, r0, r1):
                    X.append(f); y.append(0); snip.append(name)

    X = np.array(X); y = np.array(y)
    np.savez(out, X=X, y=y, snippet=np.array(snip))
    print(f'[{pid}] snippets {len(snip_idx)} | reading windows {int((y == 1).sum())} | rest windows {int((y == 0).sum())} -> {out}', flush=True)

print('done')