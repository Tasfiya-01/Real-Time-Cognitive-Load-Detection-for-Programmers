import glob, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, mne
root = r'C:\Users\HP\Downloads\Participant'
for d in sorted(glob.glob(os.path.join(root, 'Participant*'))):
    fifs = glob.glob(os.path.join(d, '*.fif'))
    csvs = glob.glob(os.path.join(d, '*_test_*.csv'))
    name = os.path.basename(d)
    if not fifs:
        print(f'{name}: no .fif yet'); continue
    try:
        raw = mne.io.read_raw_fif(fifs[0], preload=False, verbose=False)
        ev = [e for e in raw.info['events'] if e.get('list') is not None]
        n_snip = sum(1 for e in ev if e['list'][2] <= 100)
        mb = os.path.getsize(fifs[0]) / 1e6
        ok = 'OK' if n_snip == 32 and raw.info['sfreq'] == 500 else 'CHECK'
        print(f'{name}: {ok} | {mb:.0f} MB | {raw.times[-1]/60:.1f} min | sfreq {raw.info["sfreq"]} | snippet markers {n_snip} | csv {len(csvs)}')
    except Exception as e:
        print(f'{name}: ERROR -> {e}')
