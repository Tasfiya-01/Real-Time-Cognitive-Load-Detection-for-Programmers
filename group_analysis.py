import glob, re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import statsmodels.formula.api as smf

s = pd.read_csv('results/summary_participants.csv', dtype={'pid': str})
print(f'participants: {len(s)}  ->', ', '.join(s.pid))

# 1) Wilcoxon on per-participant rho
for k in ['cc_max', 'read_dur_s']:
    r = s[f'rho_{k}'].values
    msg = f'[1] rho_{k}: median {np.median(r):+.3f}, positive {int((r > 0).sum())}/{len(r)}'
    if len(r) >= 5:
        msg += f', Wilcoxon p (one-sided) {wilcoxon(r, alternative="greater").pvalue:.3f}, p (two-sided) {wilcoxon(r).pvalue:.3f}'
    else:
        msg += '  (need >= 5 participants for Wilcoxon)'
    print(msg)

# 2) pooled: z-score score within each participant, average per snippet
rows = []
for f in sorted(glob.glob('results/p*_snippets.csv')):
    d = pd.read_csv(f)
    d['pid'] = re.search(r'p(\d+)_snippets', f).group(1)          # FIX: correct participant id
    d['z'] = (d.score - d.score.mean()) / d.score.std()
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
print('participants in pooled data:', sorted(D.pid.unique()))
avg = D.groupby('snippet').agg(z=('z', 'mean'), n=('z', 'size'), cc_max=('cc_max', 'first'), cc_sum=('cc_sum', 'first')).reset_index()
for k in ['cc_max', 'cc_sum']:
    r, p = spearmanr(avg[k], avg.z)
    print(f'[2] pooled mean z-score vs {k}: rho {r:+.3f}, p {p:.3f}  ({len(avg)} snippets)')

# 3) OLS with participant-clustered standard errors (scores already z-scored per participant)
m = smf.ols('z ~ cc_max', D).fit(cov_type='cluster', cov_kwds={'groups': D['pid'].astype('category').cat.codes})
print(f'[3] OLS clustered by participant: slope of cc_max {m.params["cc_max"]:+.3f}, p {m.pvalues["cc_max"]:.3f}  (n = {len(D)}, clusters = {D.pid.nunique()})')
if D.pid.nunique() < 10:
    print('    note: clustered p-values are unreliable with fewer than ~10 participants; trust [2] more for now')
avg.to_csv('results/pooled_by_snippet.csv', index=False)
