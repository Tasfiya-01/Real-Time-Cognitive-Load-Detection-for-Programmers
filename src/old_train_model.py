import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ===== STEP 1: Load Data =====
print("Step 1: Loading data...")
X = np.load('data/X_features.npy')  # (1350, 126)
y = np.load('data/y_labels.npy')    # (1350,)

# Subject info — প্রতি subject এ 30 segments আছে
SEGMENTS_PER_SUBJECT = 30
N_SUBJECTS = 45

print(f"Total samples: {X.shape[0]}")
print(f"Subjects: {N_SUBJECTS}")

# ===== STEP 2: Subject-wise Split =====
print("\nStep 2: Subject-wise train/test split...")

# Subject index তৈরি করুন
subject_ids = np.repeat(np.arange(N_SUBJECTS), SEGMENTS_PER_SUBJECT)

# শেষ 9 subjects = test (20%)
# বাকি 36 subjects = train (80%)
test_subjects  = np.arange(36, 45)   # 9 subjects
train_subjects = np.arange(0, 36)    # 36 subjects

train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
test_idx  = np.where(np.isin(subject_ids, test_subjects))[0]

X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]

print(f"Train: {X_train.shape} — {len(train_subjects)} subjects")
print(f"Test:  {X_test.shape} — {len(test_subjects)} subjects")

# ===== STEP 3: Normalize =====
print("\nStep 3: Normalizing...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ===== STEP 4: SVM Train =====
print("\nStep 4: Training SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale',
          random_state=42, probability=True)
svm.fit(X_train, y_train)
print("Training done!")

# ===== STEP 5: Evaluate =====
print("\nStep 5: Evaluating...")
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"  SVM Accuracy: {acc*100:.2f}%")
print("="*50)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Low', 'Medium', 'High']
))

# ===== STEP 6: Leave-One-Subject-Out CV =====
print("\nStep 6: Subject-wise Cross Validation...")
cv_scores = []

for test_sub in range(N_SUBJECTS):
    # একটা subject test, বাকি সব train
    test_i  = np.where(subject_ids == test_sub)[0]
    train_i = np.where(subject_ids != test_sub)[0]

    X_tr = X[train_i]
    X_te = X[test_i]
    y_tr = y[train_i]
    y_te = y[test_i]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    clf.fit(X_tr, y_tr)
    score = accuracy_score(y_te, clf.predict(X_te))
    cv_scores.append(score)
    print(f"  Subject {test_sub+1:02d}: {score*100:.1f}%")

cv_scores = np.array(cv_scores)
print(f"\nMean CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std: {cv_scores.std()*100:.2f}%")

# ===== STEP 7: Save =====
print("\nStep 7: Saving model...")
joblib.dump(svm, 'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Saved!")

# ===== STEP 8: Plot =====
print("\nStep 8: Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cmap='Blues')
axes[0].set_title(f'Confusion Matrix\nAccuracy: {acc*100:.1f}%')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# CV Scores per subject
axes[1].bar(range(1, N_SUBJECTS+1), cv_scores*100,
            color='steelblue', alpha=0.7)
axes[1].axhline(y=cv_scores.mean()*100,
                color='red', linestyle='--',
                label=f'Mean: {cv_scores.mean()*100:.1f}%')
axes[1].set_title('Leave-One-Subject-Out CV')
axes[1].set_xlabel('Subject')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_results.png', dpi=150)
plt.show()
print("Plot saved!")

print(f"\npython -c "import src.backend" Complete!")
print(f"Test Accuracy:   {acc*100:.2f}%")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")