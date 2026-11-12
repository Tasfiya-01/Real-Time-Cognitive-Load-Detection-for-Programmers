import numpy as np
S = np.load('data/X_features_v3_fixed.npy')
F = np.load('features_p01.npz')['X']
keep = np.load('models/keep_idx.npy')
FN = ['delta','theta','alpha','beta','gamma','t/a','b/a','t/b','mean','std','var','skew','kurt','rms','sampEn','permEn','specEn','svdEn']
print('shape STEW', S.shape, 'Peitek', F.shape, 'NaN in Peitek:', int(np.isnan(F).sum()), '| kept features:', len(keep))

lo, hi = np.percentile(S, 1, axis=0), np.percentile(S, 99, axis=0)
out = ((F < lo) | (F > hi)).mean(0)
print('fraction outside, ALL columns :', round(float(out.mean()), 3))
print('fraction outside, KEPT columns:', round(float(out[keep].mean()), 3), '(~0.02 normal)')

print('\nper feature type (kept columns only), STEW median vs Peitek median, fraction outside:')
for j, n in enumerate(FN):
    cols = [c*18 + j for c in range(14) if c*18 + j in set(keep)]
    if not cols: continue
    print(f'  {n:7s} STEW {np.median(S[:, cols]):10.4f} | Peitek {np.median(F[:, cols]):10.4f} | outside {out[cols].mean():.2f}')
