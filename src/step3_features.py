"""Step 3: build NEW scale-free features from data/X_processed.npy (bandpass + z-score, shape (1350,640,14)).
Saves data/X_new_hjorth_dwt.npy (Hjorth mobility/complexity + relative DWT energy + wavelet entropy) and
data/X_new_asym.npy (left-right log band-power asymmetry).  Run:  python src/step3_features.py"""
import numpy as np, pywt
from scipy.signal import welch
from common import *

FS = 128
PAIRS = [("AF3", "AF4"), ("F3", "F4"), ("F7", "F8"), ("FC5", "FC6"), ("T7", "T8"), ("P7", "P8"), ("O1", "O2")]
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def hjorth_mob_comp(x):                        # Activity omitted: it is just the variance (amplitude)
    dx, ddx = np.diff(x), np.diff(x, n=2)
    mob = np.sqrt(np.var(dx) / (np.var(x) + 1e-12))
    comp = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (mob + 1e-12)
    return [mob, comp]


def dwt_rel(x):                                # 128 Hz, db4, 4 levels: cA4 0-4 | cD4 4-8 | cD3 8-16 | cD2 16-32 | cD1 32-64
    e = np.array([np.sum(c ** 2) for c in pywt.wavedec(x, "db4", level=4)])
    p = e / (e.sum() + 1e-12)
    return p.tolist() + [float(-np.sum(p * np.log(p + 1e-12)))]


def band_log_power(seg):
    """(n, 14, 3) log mean PSD in theta/alpha/beta (Welch, 256-sample windows, as in the thesis)."""
    f, P = welch(seg, fs=FS, nperseg=256, axis=1)             # P: (n, freqs, 14)
    out = []
    for lo, hi in BANDS.values():
        idx = (f >= lo) & (f < hi)
        out.append(np.log(P[:, idx, :].mean(1) + 1e-12))
    return np.stack(out, axis=2)


def main():
    seg = np.load(DATA / "X_processed.npy").astype(np.float64)
    assert seg.shape == (N_SUB * SEG, 640, 14), seg.shape
    F = np.array([[v for c in range(14) for v in hjorth_mob_comp(seg[i, :, c]) + dwt_rel(seg[i, :, c])]
                  for i in range(len(seg))])
    np.save(DATA / "X_new_hjorth_dwt.npy", F)
    blp = band_log_power(seg)
    A = np.column_stack([blp[:, CH.index(r), b] - blp[:, CH.index(l), b]
                         for l, r in PAIRS for b in range(len(BANDS))])
    np.save(DATA / "X_new_asym.npy", A)
    print("saved hjorth+dwt:", F.shape, "| asymmetry:", A.shape)


if __name__ == "__main__":
    main()
