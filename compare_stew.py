import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.io as sio, mne

d = sio.loadmat(r'data\dataset.mat')['dataset']        # (14, 19200, 45)
print('zero samples (fraction):', round((d == 0).mean(), 4))

rs, mx = [], []
for i in range(d.shape[2]):
    x = d[:, :, i].astype(float)
    x = mne.filter.filter_data(x, 128, 1, 45, verbose=False)   # same filter as Peitek
    x = x[:, 256:-256]                                          # drop 2 s at each edge
    rs.append(1.4826*np.median(np.abs(x - np.median(x, axis=1, keepdims=True))))
    mx.append(np.abs(x).max())

print('STEW robust std: min %.1f median %.1f max %.1f' % (min(rs), np.median(rs), max(rs)))
print('STEW recordings with max abs > 500:', sum(m > 500 for m in mx), 'of', len(mx))
print('Peitek (for comparison): robust std min 127.4 median 150.8 max 215.6, max abs > 500: 32 of 32')
