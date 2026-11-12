"""Correct vs wrong answers: (1) accuracy, (2) is cc a valid difficulty measure, (3) is EEG different on wrong answers?
Run from the project root:  python behav_analysis.py
"""
import sys, os, glob, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from xgboost import XGBClassifier
from common import N_EST, SEED

ROOT = r'C:\Users\HP\Downloads\Participant'
CC = pd.read_csv('snippet_complexity.csv').set_index('snippet')
LOWER = {n.lower(): n for n in CC.index}

# ---------- behaviour from the PsychoPy CSVs (authors' column names) ----------
rows, unmatched = [], set()
for folder in sorted(glob.glob(os.path.join(ROOT, 'Participant*'))):
    pid = re.search(r'Participant(\d+)', folder).group(1).zfill(2)
    csvs = glob.glob(os.path.join(folder, '*_test_*.csv'))
    if pid == '09' or not csvs:
        continue
    d = pd.read_csv(csvs[0])
    if 'ChoosenAnwer' not in d.columns:
        print(f'[{pid}] column ChoosenAnwer not found. Columns: {list(d.columns)[:40]}'); continue
    d = d[d['ImagePath'].notna()]
    for _, r in d.iterrows():
        name = os.path.splitext(os.path.basename(str(r['ImagePath']).replace('\\', '/')))[0]
        snip = LOWER.get(name.lower())
        if snip is None:
            unmatched.add(name); continue
        ans = str(r['ChoosenAnwer'])
        rows.append(dict(pid=pid, snippet=snip, correct=int('Right' in ans), skipped=int('Skipped' in ans)))
B = pd.DataFrame(rows)
if unmatched:
    print('snippet names not matched (check these):', sorted(unmatched)[:10])
B = B[B.skipped == 0]

print('=== (1) accuracy per participant ===')
acc = B.groupby('pid').agg(n=('correct', 'size'), correct=('correct', 'sum'))
acc['accuracy'] = (acc.correct / acc.n).round(3)
print(acc.to_string())
print(f'overall accuracy: {B.correct.mean():.3f}')

# ---------- (2) is cc a valid difficulty measure? ----------
times = pd.concat([pd.read_csv(f).assign(pid=re.search(r'p(\d+)_snippets', f).group(1))
                   for f in glob.glob('results/p*_snippets.csv')])
times['t_z'] = times.groupby('pid').read_dur_s.transform(lambda v: (v - v.mean()) / v.std())
per_snip = B.groupby('snippet').correct.mean().rename('acc').to_frame() \
    .join(times.groupby('snippet').t_z.mean().rename('time_z')).join(CC[['cc_max', 'cc_sum', 'nloc']])
print('\n=== (2) does cc predict behaviour? (32 snippets, pooled over participants) ===')
for k in ['cc_max', 'cc_sum', 'nloc']:
    r1, p1 = spearmanr(per_snip[k], 1 - per_snip.acc)
    r2, p2 = spearmanr(per_snip[k], per_snip.time_z)
    print(f'  {k:7s} vs error rate: rho {r1:+.3f} (p {p1:.3f}) | vs reading time: rho {r2:+.3f} (p {p2:.3f})')
print('  hardest snippets by error rate:', ', '.join((1 - per_snip.acc).sort_values(ascending=False).head(5).round(2).astype(str).radd(per_snip.acc.sort_values().head(5).index + ' ')))

# ---------- (3) EEG on wrong vs correct answers ----------
# STEW model score per snippet (already computed)
stew = times[['pid', 'snippet', 'score']].rename(columns={'score': 'stew'})
# programmer model (LOSO, reading vs rest) -> mean "reading-likeness" per snippet
keep = np.load('models/keep_idx.npy')
data = {re.search(r'rr_p(\d+)', f).group(1): np.load(f) for f in sorted(glob.glob('results/rr_p*.npz'))}
prog = []
for test in data:
    Xtr = np.vstack([data[p]['X'][:, keep] for p in data if p != test])
    ytr = np.concatenate([data[p]['y'] for p in data if p != test])
    w = np.where(ytr == 1, 0.5 / ytr.mean(), 0.5 / (1 - ytr.mean()))
    m = XGBClassifier(n_estimators=N_EST, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                      random_state=SEED, n_jobs=-1).fit(Xtr, ytr, sample_weight=w)
    X, y, s = data[test]['X'][:, keep], data[test]['y'], data[test]['snippet']
    p = m.predict_proba(X[y == 1])[:, 1]
    g = pd.DataFrame({'snippet': s[y == 1], 'prog': p}).groupby('snippet').prog.mean().reset_index()
    g['pid'] = test
    prog.append(g)
prog = pd.concat(prog)

M = B.merge(stew, on=['pid', 'snippet']).merge(prog, on=['pid', 'snippet'])
print('\n=== (3) EEG score on WRONG minus CORRECT answers (within each participant) ===')
res = []
for pid, g in M.groupby('pid'):
    nw, nc = int((g.correct == 0).sum()), int((g.correct == 1).sum())
    if nw >= 2 and nc >= 2:
        res.append(dict(pid=pid, n_wrong=nw, n_correct=nc,
                        stew_diff=g[g.correct == 0].stew.mean() - g[g.correct == 1].stew.mean(),
                        prog_diff=g[g.correct == 0].prog.mean() - g[g.correct == 1].prog.mean()))
R = pd.DataFrame(res)
if len(R):
    print(R.round(3).to_string(index=False))
    for col in ['stew_diff', 'prog_diff']:
        msg = f'  {col}: median {R[col].median():+.3f}, positive {int((R[col] > 0).sum())}/{len(R)}'
        if len(R) >= 5:
            msg += f', Wilcoxon p = {wilcoxon(R[col]).pvalue:.3f}'
        print(msg)
else:
    print('  not enough participants with >= 2 wrong and >= 2 correct answers')
B.to_csv('results/behaviour.csv', index=False)
print('\nsaved results/behaviour.csv')