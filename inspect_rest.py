import numpy as np, pandas as pd, mne, warnings
warnings.filterwarnings("ignore")

p = r'C:\Users\HP\Downloads\Participant\Participant01'
raw = mne.io.read_raw_fif(p + r'\CollectedData_2021-07-29_1.fif', preload=False, verbose=False)
ev = np.array([[e['list'][2], e['list'][0]] for e in raw.info['events'] if e.get('list') is not None])
t = ev[:, 1] / raw.info['sfreq']

print('first 14 events (id, time s, gap to next s):')
for i in range(14):
    print(f'  {int(ev[i,0]):>12}  {t[i]:8.2f}  {t[i+1]-t[i]:7.2f}')

d = pd.read_csv(p + r'\001_test_2021_Jul_29_1501.csv')
cols = [c for c in d.columns if 'image_7' in c or 'cross' in c.lower() or 'Image.st' in c]
print('\nPsychoPy columns about cross/snippet:', cols)
print(d[cols].dropna(how='all').head(4).to_string())