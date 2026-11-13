"""E5: Theta/alpha load index on confusing vs clean code (Bergum et al., Atoms of Confusion EEG data).

Markers in the BrainVision files (from the authors' eeg_settings.py):
  S  3 = fixation cross (about 5 s, before each snippet)  -> baseline
  S 11 = clean snippet start, S 12 = confusing snippet start
  S  4 = snippet end (answer given)
Usage:  python aoc_index.py <folder with .vhdr files>
Outputs: results/aoc_trials.csv, results/aoc_summary.txt, results/fig_aoc_index.png
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, mne
from scipy.signal import welch
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

mne.set_log_level("ERROR")
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "aoc_data")
OUT = Path("results"); OUT.mkdir(exist_ok=True)
CHS = ["Fz", "Pz", "F3", "F4", "P7", "P8"]
WIN, STEP = 3.0, 1.0          # same as E3 in the thesis
THETA, ALPHA, TOTAL = (4, 8), (8, 13), (1, 40)

def code(desc):
    m = re.search(r"S\s*(\d+)", desc)
    return int(m.group(1)) if m else None

def rel_power(x, fs, band):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    tot = p[(f >= TOTAL[0]) & (f < TOTAL[1])].sum()
    return p[(f >= band[0]) & (f < band[1])].sum() / tot

def index_series(data, fs, t0, t1, ch):
    """log10(rel theta at frontal / rel alpha at parietal), 3-s windows every 1 s, mean over windows."""
    vals_a, vals_e = [], []
    t = t0
    while t + WIN <= t1 + 1e-9:
        a, b = int(t * fs), int((t + WIN) * fs)
        seg = data[:, a:b]
        th = lambda c: rel_power(seg[ch.index(c)], fs, THETA)
        al = lambda c: rel_power(seg[ch.index(c)], fs, ALPHA)
        vals_a.append(np.log10(th("Fz") / al("Pz")))
        vals_e.append(np.log10(((th("F3") + th("F4")) / 2) / ((al("P7") + al("P8")) / 2)))
        t += STEP
    if not vals_a:
        return np.nan, np.nan
    return float(np.mean(vals_a)), float(np.mean(vals_e))

rows = []
files = sorted(ROOT.rglob("*.vhdr"))
print("vhdr files found:", len(files))
for f in files:
    try:
        raw = mne.io.read_raw_brainvision(f, preload=False)
    except Exception as e:
        print("skip", f.name, e); continue
    missing = [c for c in CHS if c not in raw.ch_names]
    if missing:
        print("skip", f.name, "missing", missing); continue
    raw.pick(CHS).load_data()
    raw.notch_filter([50, 100], verbose=False)
    raw.filter(1, 40, verbose=False)
    fs = raw.info["sfreq"]; data = raw.get_data(); ch = raw.ch_names
    ann = [(o, code(d)) for o, d in zip(raw.annotations.onset, raw.annotations.description) if code(d) is not None]
    pid = f.parent.name if f.parent != ROOT else f.stem
    trial = 0
    for i, (on, c) in enumerate(ann):
        if c not in (11, 12):
            continue
        end = next((o for o, cc in ann[i + 1:] if cc == 4), None)
        fix = next((o for o, cc in reversed(ann[:i]) if cc == 3), None)
        if end is None or end - on < WIN:
            continue
        trial += 1
        r_a, r_e = index_series(data, fs, on, end, ch)
        b_a, b_e = (index_series(data, fs, fix, on, ch) if fix is not None and on - fix >= WIN else (np.nan, np.nan))
        rows.append(dict(pid=pid, file=f.name, trial=trial, condition="confusing" if c == 12 else "clean",
                         read_dur=end - on, base_dur=(on - fix) if fix is not None else np.nan,
                         read_fzpz=r_a, read_emotiv=r_e, base_fzpz=b_a, base_emotiv=b_e))
    print(f"{f.name}: pid={pid}, trials={trial}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "aoc_trials.csv", index=False)
if df.empty:
    sys.exit("No trials found - send me the printed output.")

# z-score each participant's reading values against that participant's own baseline (fixation) windows
for v in ("fzpz", "emotiv"):
    def z(g):
        mu, sd = g[f"base_{v}"].mean(), g[f"base_{v}"].std(ddof=1)
        return (g[f"read_{v}"] - mu) / sd
    df[f"z_{v}"] = df.groupby("pid", group_keys=False).apply(z)

lines = []
def say(s=""):
    print(s); lines.append(s)

say(f"Participants: {df.pid.nunique()}, trials: {len(df)} (clean {sum(df.condition=='clean')}, confusing {sum(df.condition=='confusing')})")
say(f"Mean reading time: clean {df[df.condition=='clean'].read_dur.mean():.1f} s, confusing {df[df.condition=='confusing'].read_dur.mean():.1f} s; baseline {df.base_dur.mean():.1f} s")
per = df.groupby(["pid", "condition"])[["z_fzpz", "z_emotiv", "read_dur"]].mean().unstack()
for v, name in (("z_fzpz", "Fz/Pz"), ("z_emotiv", "F3F4/P7P8")):
    read_vs_rest = df.groupby("pid")[v].mean().dropna()
    d = (per[(v, "confusing")] - per[(v, "clean")]).dropna()
    say(f"\n[{name}] reading vs fixation baseline: mean z {read_vs_rest.mean():+.3f}, "
        f"{(read_vs_rest > 0).sum()}/{len(read_vs_rest)} positive, Wilcoxon p = {wilcoxon(read_vs_rest).pvalue:.4f}")
    say(f"[{name}] CONFUSING minus CLEAN: mean {d.mean():+.3f} z, {(d > 0).sum()}/{len(d)} participants higher on confusing, "
        f"Wilcoxon p = {wilcoxon(d).pvalue:.4f}")
dt = (per[("read_dur", "confusing")] - per[("read_dur", "clean")]).dropna()
say(f"\nReading time confusing minus clean: {dt.mean():+.2f} s, {(dt > 0).sum()}/{len(dt)} longer on confusing, p = {wilcoxon(dt).pvalue:.4f}")
(OUT / "aoc_summary.txt").write_text("\n".join(lines), encoding="utf-8")

# figure: per participant, confusing - clean (Fz/Pz and Emotiv version)
dd = pd.DataFrame({"Fz/Pz": per[("z_fzpz", "confusing")] - per[("z_fzpz", "clean")],
                   "F3F4/P7P8": per[("z_emotiv", "confusing")] - per[("z_emotiv", "clean")]}).dropna()
ax = dd.plot.bar(figsize=(12, 4.5), color=["#2b5d8a", "#8fb3d9"], edgecolor="black", linewidth=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Load index, confusing − clean (z)")
ax.set_title("Theta/alpha load index: confusing minus clean code, per participant")
plt.tight_layout(); plt.savefig(OUT / "fig_aoc_index.png", dpi=180)
print("\nSaved results/aoc_trials.csv, results/aoc_summary.txt, results/fig_aoc_index.png")