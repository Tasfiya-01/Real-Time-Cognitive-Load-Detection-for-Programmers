import numpy as np, antropy as ant

FS = 128
X_feat = np.load("data/X_features_v3.npy")
X_proc = np.load("data/X_processed.npy")          # (1350, 640, 14)
assert X_feat.shape[0] == X_proc.shape[0] and X_feat.shape[1] == 252

X_new = X_feat.copy()
n, _, n_ch = X_proc.shape
for i in range(n):
    for ch in range(n_ch):
        s = np.ascontiguousarray(X_proc[i, :, ch], dtype=np.float64)
        base = ch * 18
        X_new[i, base + 14] = ant.sample_entropy(s)
        X_new[i, base + 15] = ant.perm_entropy(s, normalize=True)
        X_new[i, base + 16] = ant.spectral_entropy(s, sf=FS, normalize=True)
        X_new[i, base + 17] = ant.svd_entropy(s, normalize=True)
    if i % 100 == 0:
        print(i, "/", n, flush=True)

# আগের ফাইল ব্যাকআপ, তারপর নতুনটা সেভ
np.save("data/X_features_v3_old.npy", X_feat)
np.save("data/X_features_v3_fixed.npy", X_new)
print("nan/inf:", int((~np.isfinite(X_new)).sum()))
print("entropy columns std (first channel):", X_new[:, 14:18].std(0).round(4))
print("saved data/X_features_v3_fixed.npy")