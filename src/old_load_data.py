import scipy.io as sio
import numpy as np

# Dataset load করুন
print("Loading dataset...")

dataset = sio.loadmat('data/dataset.mat')
labels  = sio.loadmat('data/class_012.mat')
rating  = sio.loadmat('data/rating.mat')

# Data বের করুন
eeg_data  = dataset['dataset']
class_labels = labels['class_012']
rating_data  = rating['rating']

# Shape দেখুন
print("\n--- Data Info ---")
print("EEG data shape:   ", eeg_data.shape)
print("Labels shape:     ", class_labels.shape)
print("Rating shape:     ", rating_data.shape)

# Labels কী কী আছে
print("\n--- Labels ---")
print("Unique labels:", np.unique(class_labels))
print("Label counts:")
for label in np.unique(class_labels):
    count = np.sum(class_labels == label)
    print(f"  Class {label}: {count} samples")