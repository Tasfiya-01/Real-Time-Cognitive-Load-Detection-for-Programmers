import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ===== STEP 1: Load Processed Data =====
print("Step 1: Loading processed data...")
X = np.load('data/X_processed.npy')  # (1350, 640, 14)
y = np.load('data/y_labels.npy')     # (1350,)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ===== STEP 2: Feature Extraction Functions =====
FS = 128

def band_power(channel, low, high, fs=FS):
    freqs, psd = welch(channel, fs=fs, nperseg=256)
    idx = np.where((freqs >= low) & (freqs < high))[0]
    return np.mean(psd[idx])

def extract_features(segment):
    """
    segment: (640, 14)
    প্রতিটা channel থেকে 9টা feature বের করব
    মোট: 14 channels × 9 features = 126 features
    """
    features = []
    
    for ch in range(segment.shape[1]):  # 14 channels
        ch_data = segment[:, ch]
        
        # Band power features (4টা)
        theta = band_power(ch_data, 4, 8)    # Mental workload
        alpha = band_power(ch_data, 8, 13)   # Relaxation
        beta  = band_power(ch_data, 13, 30)  # Active thinking
        gamma = band_power(ch_data, 30, 50)  # Complex thinking
        
        # Statistical features (5টা)
        mean = np.mean(ch_data)
        std  = np.std(ch_data)
        var  = np.var(ch_data)
        sk   = skew(ch_data)
        kurt = kurtosis(ch_data)
        
        features.extend([theta, alpha, beta, gamma,
                         mean, std, var, sk, kurt])
    
    return np.array(features)

# ===== STEP 3: Extract All Features =====
print("\nStep 2: Extracting features...")


X_features = []
for i, seg in enumerate(X):
    feat = extract_features(seg)
    X_features.append(feat)
    if (i+1) % 100 == 0:
        print(f"  Progress: {i+1}/{len(X)} done...")

X_features = np.array(X_features)
print(f"\nFeature matrix shape: {X_features.shape}")
print(f"Features per sample: {X_features.shape[1]}")

# ===== STEP 4: Save Features =====
print("\nStep 3: Saving features...")
np.save('data/X_features.npy', X_features)
np.save('data/y_labels.npy', y)
print("Saved! X_features.npy")

# ===== STEP 5: Quick Check =====
print("\n--- Feature Summary ---")
print(f"Total samples:     {X_features.shape[0]}")
print(f"Features per sample: {X_features.shape[1]}")
print(f"Class 0 (Low):     {sum(y==0)}")
print(f"Class 1 (Medium):  {sum(y==1)}")
print(f"Class 2 (High):    {sum(y==2)}")
print("\n Feature extraction complete!")