import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ===== STEP 1: Load Processed Data =====
print("Step 1: Loading data...")
X_raw = np.load('data/X_processed.npy')  # (1350, 640, 14)
y     = np.load('data/y_labels.npy')     # (1350,)
print(f"X shape: {X_raw.shape}")

# ===== STEP 2: Better Feature Extraction =====
print("\nStep 2: Extracting better features...")
from scipy.signal import welch
from scipy.stats import skew, kurtosis

FS = 128

def extract_features_v2(segment):
    """
    segment: (640, 14)
    আরো বেশি features বের করব
    """
    features = []

    for ch in range(segment.shape[1]):
        ch_data = segment[:, ch]
        freqs, psd = welch(ch_data, fs=FS, nperseg=256)

        def bp(low, high):
            idx = np.where((freqs >= low) & (freqs < high))[0]
            return np.mean(psd[idx])

        # Band powers
        delta = bp(0.5, 4)
        theta = bp(4, 8)
        alpha = bp(8, 13)
        beta  = bp(13, 30)
        gamma = bp(30, 50)

        # Ratios — cognitive load এর জন্য গুরুত্বপূর্ণ
        theta_alpha = theta / (alpha + 1e-8)
        beta_alpha  = beta  / (alpha + 1e-8)
        theta_beta  = theta / (beta  + 1e-8)

        # Statistical
        mean = np.mean(ch_data)
        std  = np.std(ch_data)
        var  = np.var(ch_data)
        sk   = skew(ch_data)
        kurt = kurtosis(ch_data)
        rms  = np.sqrt(np.mean(ch_data**2))

        features.extend([
            delta, theta, alpha, beta, gamma,
            theta_alpha, beta_alpha, theta_beta,
            mean, std, var, sk, kurt, rms
        ])

    return np.array(features)

X_features = []
for i, seg in enumerate(X_raw):
    feat = extract_features_v2(seg)
    X_features.append(feat)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(X_raw)} done...")

X_features = np.array(X_features)
print(f"Feature matrix: {X_features.shape}")

# ===== STEP 3: Simple Train/Test Split =====
print("\nStep 3: Splitting data...")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

# ===== STEP 4: Multiple Models Try করুন =====
print("\nStep 4: Training models...")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    'SVM':            SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'Random Forest':  RandomForestClassifier(n_estimators=200, random_state=42),
    'KNN':            KNeighborsClassifier(n_neighbors=5),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
best_acc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name} Accuracy: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# ===== STEP 5: Best Model Details =====
print("\n" + "="*50)
print(f"Best Model: {best_name}")
print(f"Best Accuracy: {best_acc*100:.2f}%")
print("="*50)

y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_best,
    target_names=['Low', 'Medium', 'High']
))

# ===== STEP 6: Save Best Model =====
print("\nStep 6: Saving best model...")
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
np.save('data/X_features_v2.npy', X_features)
print(f"Saved! Best model: {best_name}")

# ===== STEP 7: Plot =====
print("\nStep 7: Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model Comparison
names = list(results.keys())
accs  = [v*100 for v in results.values()]
colors = ['steelblue' if n != best_name else 'green' for n in names]
axes[0].bar(names, accs, color=colors)
axes[0].set_title('Model Comparison')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([0, 100])
for i, (n, a) in enumerate(zip(names, accs)):
    axes[0].text(i, a+1, f'{a:.1f}%', ha='center', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cmap='Blues')
axes[1].set_title(f'Best Model: {best_name}\nAccuracy: {best_acc*100:.1f}%')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=150)
plt.show()
print("Plot saved!")

print(f"\n Complete! Best: {best_name} — {best_acc*100:.2f}%")