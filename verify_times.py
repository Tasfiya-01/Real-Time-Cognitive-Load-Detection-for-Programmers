import sys, numpy as np, pandas as pd, mne
fif = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\HP\Downloads\CollectedData_2021-07-29_1.fif'
raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
fs = raw.info['sfreq']

# same logic as the authors' preparers.py (get_events_from_info)
ev = np.array([[e['list'][2], e['list'][1], e['list'][0]] for e in raw.info['events'] if e.get('list') is not None])
ids, t = ev[:, 0], ev[:, 2] / fs
rows = [(int(ids[i]), t[i+1], t[i+2]) for i in range(len(ids)) if ids[i] <= 100 and i + 2 < len(ids)]
auth = pd.DataFrame(rows, columns=['snip_id', 'start_auth', 'end_auth'])
print('events in info:', len(ev), '| snippet markers found:', len(auth))

mine = pd.read_csv('snippet_eeg_times.csv')
n = min(len(mine), len(auth))
cmp = pd.concat([mine[['order', 'snippet', 'read_start_s', 'read_end_s']].iloc[:n].reset_index(drop=True), auth.iloc[:n]], axis=1)
cmp['d_start'] = cmp.read_start_s - cmp.start_auth
cmp['d_end'] = cmp.read_end_s - cmp.end_auth
print(cmp.round(2).to_string(index=False))
print('\nmax |difference| start: %.3f s   end: %.3f s' % (cmp.d_start.abs().max(), cmp.d_end.abs().max()))
