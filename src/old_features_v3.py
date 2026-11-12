import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import antropy as ant

FS = 128

def extract_features_v3(segment):
    features = []

    for ch in range(segment.shape[1]):
        ch_data = np.ascontiguousarray(segment[:, ch], dtype=np.float64)
        freqs, psd = welch(ch_data, fs=FS, nperseg=256)

        def bp(low, high):
            idx = np.where((freqs >= low) & (freqs < high))[0]
            return np.mean(psd[idx])

        delta = bp(0.5, 4)
        theta = bp(4, 8)
        alpha = bp(8, 13)
        beta  = bp(13, 30)
        gamma = bp(30, 50)

        theta_alpha = theta / (alpha + 1e-8)
        beta_alpha  = beta  / (alpha + 1e-8)
        theta_beta  = theta / (beta  + 1e-8)

        mean = np.mean(ch_data)
        std  = np.std(ch_data)
        var  = np.var(ch_data)
        sk   = skew(ch_data)
        kurt = kurtosis(ch_data)
        rms  = np.sqrt(np.mean(ch_data**2))

        try:
            samp_en  = ant.sample_entropy(ch_data)
            perm_en  = ant.perm_entropy(ch_data, normalize=True)
            spec_en  = ant.spectral_entropy(ch_data, sf=FS, normalize=True)
            svd_en   = ant.svd_entropy(ch_data, normalize=True)
        except Exception as e:
            raise RuntimeError(f"entropy failed: {e}")

        features.extend([
            delta, theta, alpha, beta, gamma,
            theta_alpha, beta_alpha, theta_beta,
            mean, std, var, sk, kurt, rms,
            samp_en, perm_en, spec_en, svd_en
        ])

    return np.array(features)


if __name__ == '__main__':
    import scipy.io as sio
    from scipy import signal as sig

    print("Loading data...")
    dataset = sio.loadmat('data/dataset.mat')
    labels  = sio.loadmat('data/class_012.mat')

    eeg_data     = dataset['dataset']
    class_labels = labels['class_012'].flatten()

    data_T = np.transpose(eeg_data, (2, 1, 0))

    SEGMENT_LEN = 640

    def bandpass_filter(segment):
        nyq = 0.5 * FS
        b, a = sig.butter(4, [0.5/nyq, 50.0/nyq], btype='band')
        filtered = np.zeros_like(segment)
        for ch in range(segment.shape[1]):
            filtered[:, ch] = sig.filtfilt(b, a, segment[:, ch])
        return filtered

    def normalize(segment):
        normalized = np.zeros_like(segment)
        for ch in range(segment.shape[1]):
            d = segment[:, ch]
            normalized[:, ch] = (d - np.mean(d)) / (np.std(d) + 1e-8)
        return normalized

    print("Segmenting...")
    segments, seg_labels = [], []
    for i in range(data_T.shape[0]):
        subject = data_T[i]
        n_seg = subject.shape[0] // SEGMENT_LEN
        for j in range(n_seg):
            seg = subject[j*SEGMENT_LEN:(j+1)*SEGMENT_LEN, :]
            seg = bandpass_filter(seg)
            seg = normalize(seg)
            segments.append(seg)
            seg_labels.append(class_labels[i])

    X_raw = np.array(segments)
    y     = np.array(seg_labels)

    print(f"Extracting entropy features... ({len(X_raw)} segments)")
    print("This will take 10-15 minutes...")

    X_features = []
    for i, seg in enumerate(X_raw):
        feat = extract_features_v3(seg)
        X_features.append(feat)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(X_raw)} done...")

    X_features = np.array(X_features)
    print(f"Feature matrix shape: {X_features.shape}")

    np.save('data/X_features_v3.npy', X_features)
    np.save('data/y_labels.npy', y)
    print("Saved! X_features_v3.npy")
