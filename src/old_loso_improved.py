import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

print("Loading data...")
X = np.load('data/X_features_v3.npy')
y = np.load('data/y_labels.npy')

N_SUBJECTS = 45
SEG_PER_SUBJECT = 30
subject_ids = np.repeat(np.arange(N_SUBJECTS), SEG_PER_SUBJECT)

def loso_evaluate(name, clf_fn):
    scores = []
    for test_sub in range(N_SUBJECTS):
        test_i  = np.where(subject_ids == test_sub)[0]
        train_i = np.where(subject_ids != test_sub)[0]

        X_tr, X_te = X[train_i], X[test_i]
        y_tr, y_te = y[train_i], y[test_i]

        # RobustScaler — outlier এ বেশি robust
        sc = RobustScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        clf = clf_fn()
        clf.fit(X_tr, y_tr)
        score = accuracy_score(y_te, clf.predict(X_te))
        scores.append(score)

    mean = np.mean(scores) * 100
    print(f"{name}: {mean:.2f}%")
    return scores

print("\nTrying different approaches...")
print("="*50)

# Method 1: XGBoost tuned
s1 = loso_evaluate("XGBoost (tuned)", lambda: XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6,
    min_child_weight=3, gamma=0.1,
    eval_metric='mlogloss', verbosity=0, random_state=42
))

# Method 2: SVM
s2 = loso_evaluate("SVM (rbf)", lambda: SVC(
    kernel='rbf', C=5, gamma='scale', random_state=42
))

# Method 3: LDA
s3 = loso_evaluate("LDA", lambda: LinearDiscriminantAnalysis(
    solver='svd', shrinkage=None
))

# Method 4: Random Forest
s4 = loso_evaluate("Random Forest", lambda: RandomForestClassifier(
    n_estimators=300, max_depth=8,
    min_samples_leaf=3, random_state=42
))

print("\n" + "="*50)
print(f"Previous LOSO:        35.00%")
print(f"Best new result:      {max(np.mean(s1), np.mean(s2), np.mean(s3), np.mean(s4))*100:.2f}%")

# ==== এইটুকু src/loso_improved.py এর একদম শেষে যোগ করুন ====
import matplotlib.pyplot as plt
import numpy as np

methods = ["XGBoost\n(tuned)", "SVM\n(rbf)", "LDA", "Random\nForest"]
results = [np.mean(s1)*100, np.mean(s2)*100, np.mean(s3)*100, np.mean(s4)*100]
best_idx = int(np.argmax(results))

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#4C72B0" if i != best_idx else "#2E8B57" for i in range(len(methods))]
bars = ax.bar(methods, results, color=colors, width=0.55)

for bar, val in zip(bars, results):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.2f}%",
            ha="center", fontsize=11, fontweight="bold")

ax.axhline(35.0, color="red", linestyle="--", linewidth=1.3, label="Original LOSO baseline (35.00%)")
ax.set_ylabel("LOSO Mean Accuracy (%)")
ax.set_title("Leave-One-Subject-Out Accuracy Across Mitigation Strategies")
ax.set_ylim(0, max(results) + 10)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/loso_mitigation_comparison.png", dpi=200)
print("Saved: results/loso_mitigation_comparison.png")