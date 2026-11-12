"""Cognitive load index (frontal theta / parietal alpha), relative to each person's own rest.
Two versions: authors' (Fz/Pz) and Emotiv-like (F3,F4 / P7,P8).
Run from the project root:  python cl_index.py
"""
import os, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, mne
from scipy.signal import welch
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt

ROOT = r'C:\Users\HP\Downloads\Participant'
CC = pd.read_csv('snippet_complexity.csv').set_index('snippet')
NAMES = sorted(CC.index, key=str.lower)
CHS = ['Fz', 'Pz', 'F3', 'F4', 'P7', 'P8']
I = {c: i for i, c in enumerate(CHS)}


def rel(f, P, lo, hi):
    """relative band power (share of 1-40 Hz power)"""
    return P[..., (f >= lo) & (f < hi)].sum(-1) / P[..., (f >= 1) & (f < 40)].sum(-1)


def load_index(seg, fs):
    """3-s windows every 1 s -> log10(theta/alpha) for both versions"""
    W, S = int(3 * fs), int(fs)
    starts = list(range(0, seg.shape[1] - W + 1, S))
    if not starts:
        return None, None
    X = np.stack([seg[:, s:s + W] for s in starts])
    f, P = welch(X, fs=fs, nperseg=int(fs), axis=-1)
    th, al = rel(f, P, 4, 8), rel(f, P, 8, 13)
    authors = np.log10(th[:, I['Fz']] / al[:, I['Pz']])
    emotiv = np.log10(th[:, [I['F3'], I['F4']]].mean(1) / al[:, [I['P7'], I['P8']]].mean(1))
    return authors, emotiv


rows = []
for folder in sorted(glob.glob(os.path.join(ROOT, 'Participant*'))):
    pid = re.search(r'Participant(\d+)', folder).group(1).zfill(2)
    fifs = glob.glob(os.path.join(folder, '*.fif'))
    if pid == '09' or not fifs:
        continue
    raw = mne.io.read_raw_fif(fifs[0], preload=False, verbose=False)
    missing = [c for c in CHS if c not in raw.ch_names]
    if missing:
        print(f'[{pid}] missing channels {missing}, skipped'); continue
    fs = raw.info['sfreq']
    ev = np.array([[e['list'][2], e['list'][0]] for e in raw.info['events'] if e.get('list') is not None])
    ids, t = ev[:, 0], ev[:, 1] / fs
    snip_idx = [i for i in range(len(ids)) if ids[i] <= 100 and i + 3 < len(ids)]

    raw.pick(CHS).load_data(); raw.reorder_channels(CHS)
    raw.notch_filter([50, 100, 150, 200], verbose=False)
    raw.filter(1, 40, verbose=False)
    data = raw.get_data()

    for n, i in enumerate(snip_idx):
        name = NAMES[int(ids[i])]
        a, e = load_index(data[:, int(t[i+1]*fs):int(t[i+2]*fs)], fs)            # reading
        if a is not None:
            rows += [dict(pid=pid, snippet=name, phase='read', authors=x, emotiv=y, read_dur=t[i+2]-t[i+1]) for x, y in zip(a, e)]
        if n + 1 < len(snip_idx):                                               # rest after this snippet
            nxt = t[snip_idx[n+1]]
            if nxt - 29 > t[i+3]:
                a, e = load_index(data[:, int((nxt-29)*fs):int((nxt-1)*fs)], fs)
                if a is not None:
                    rows += [dict(pid=pid, snippet=name, phase='rest', authors=x, emotiv=y, read_dur=np.nan) for x, y in zip(a, e)]
    print(f'[{pid}] done', flush=True)

D = pd.DataFrame(rows)
# z-score against each person's own rest (rest -> about 0, positive = more load than rest)
for v in ['authors', 'emotiv']:
    st = D[D.phase == 'rest'].groupby('pid')[v].agg(['mean', 'std'])
    D[v + '_z'] = (D[v] - D.pid.map(st['mean'])) / D.pid.map(st['std'])
D.to_csv('results/cl_index_windows.csv', index=False)

# (1) load during code reading vs rest
print('\n=== (1) cognitive load during CODE READING relative to REST (0 = rest level) ===')
P1 = D[D.phase == 'read'].groupby('pid')[['authors_z', 'emotiv_z']].mean()
print(P1.round(3).to_string())
for v in ['authors_z', 'emotiv_z']:
    print(f'  {v:10s}: mean {P1[v].mean():+.3f}, higher than rest in {int((P1[v] > 0).sum())}/{len(P1)}, Wilcoxon p = {wilcoxon(P1[v]).pvalue:.4f}')

# (2) + (3) per-snippet load vs code complexity and reading time
S = D[D.phase == 'read'].groupby(['pid', 'snippet']).agg(authors_z=('authors_z', 'mean'), emotiv_z=('emotiv_z', 'mean'),
                                                         read_dur=('read_dur', 'first')).reset_index()
S = S.join(CC[['cc_max', 'nloc']], on='snippet')
print('\n=== (2)(3) per-snippet load vs complexity / reading time ===')
for v in ['authors_z', 'emotiv_z']:
    for k in ['cc_max', 'nloc', 'read_dur']:
        r = S.groupby('pid').apply(lambda g: spearmanr(g[k], g[v])[0])
        pooled = S.groupby('snippet')[[v, k]].mean()
        pr, pp = spearmanr(pooled[k], pooled[v])
        print(f'  {v:10s} vs {k:8s}: per-person median rho {r.median():+.3f} (positive {int((r > 0).sum())}/{len(r)}, '
              f'Wilcoxon p {wilcoxon(r).pvalue:.3f}) | pooled rho {pr:+.3f}, p {pp:.3f}')

# figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(P1)); w = 0.38
ax[0].bar(x - w/2, P1.authors_z, w, color='#1d4ed8', label=f'authors (Fz/Pz), mean {P1.authors_z.mean():+.2f}')
ax[0].bar(x + w/2, P1.emotiv_z, w, color='#60a5fa', label=f'Emotiv-like (F3F4/P7P8), mean {P1.emotiv_z.mean():+.2f}')
ax[0].axhline(0, color='black', lw=1); ax[0].text(-0.6, 0.03, 'rest level', fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels('P' + P1.index)
ax[0].set_ylabel('cognitive load index during code reading\n(z relative to own rest)')
ax[0].set_title('Cognitive load (theta/alpha) during code comprehension vs rest')
ax[0].legend(fontsize=8)
pooled = S.groupby('snippet')[['authors_z', 'cc_max']].mean()
ax[1].scatter(pooled.cc_max, pooled.authors_z, color='#1d4ed8')
r, p = spearmanr(pooled.cc_max, pooled.authors_z)
ax[1].set_xlabel('cyclomatic complexity (cc_max)'); ax[1].set_ylabel('mean load index (z), all participants')
ax[1].set_title(f'Load per snippet vs complexity (rho {r:+.2f}, p {p:.2f})')
plt.tight_layout(); plt.savefig('results/fig_load_index.png', dpi=200, bbox_inches='tight')
print('\nsaved results/fig_load_index.png and results/cl_index_windows.csv')