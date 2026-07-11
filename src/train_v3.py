import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading features...")
X = np.load('data/X_features_v3.npy')
y = np.load('data/y_labels.npy')
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

models = {
    'SVM':            SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True),
    'KNN':            KNeighborsClassifier(n_neighbors=5),
    'Random Forest':  RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
best_acc = 0
best_model = None
best_name = ""

print("\nTraining models...")
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name} Accuracy: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

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

print("\nCross Validation (5-fold)...")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")

print("\nSaving best model...")
joblib.dump(best_model, 'models/best_model_v3.pkl')
joblib.dump(scaler, 'models/scaler_v3.pkl')
print("Saved!")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
accs  = [v*100 for v in results.values()]
colors = ['green' if n == best_name else 'steelblue' for n in names]
axes[0].bar(names, accs, color=colors)
axes[0].set_title('Model Comparison (With Entropy Features)')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([0, 100])
for i, (n, a) in enumerate(zip(names, accs)):
    axes[0].text(i, a+1, f'{a:.1f}%', ha='center', fontsize=11)
axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cmap='Blues')
axes[1].set_title(f'Best Model: {best_name}\nAccuracy: {best_acc*100:.1f}%')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/model_v3_results.png', dpi=150)
plt.show()
print("Plot saved!")

print(f"\nDone! Best: {best_name} — {best_acc*100:.2f}%")
print(f"Previous accuracy: 77.78%")
print(f"New accuracy:      {best_acc*100:.2f}%")
improvement = best_acc*100 - 77.78
print(f"Improvement:       +{improvement:.2f}%")