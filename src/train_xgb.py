import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading features...")
X = np.load('data/X_features_v3.npy')
y = np.load('data/y_labels.npy')
print(f"X shape: {X.shape}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")

print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"XGBoost Accuracy: {acc*100:.2f}%")
print("="*50)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Low', 'Medium', 'High']
))

print("\nCross Validation (5-fold)...")
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")

print("\nSaving model...")
joblib.dump(xgb, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler_v3.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("Saved!")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models_compare = {
    'Original KNN\n(v1)': 77.78,
    'Random Forest\n(v2)': 78.52,
    f'XGBoost\n(v3)': acc*100
}
colors = ['steelblue', 'steelblue', 'green']
axes[0].bar(models_compare.keys(), models_compare.values(), color=colors)
axes[0].set_title('Accuracy Improvement Over Versions')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([70, 100])
for i, (n, a) in enumerate(models_compare.items()):
    axes[0].text(i, a+0.3, f'{a:.2f}%', ha='center', fontsize=11)
axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cmap='Blues')
axes[1].set_title(f'XGBoost Confusion Matrix\nAccuracy: {acc*100:.1f}%')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/xgb_results.png', dpi=150)
plt.show()
print("Plot saved!")

print(f"\nSummary:")
print(f"Original paper (SVM):  69.00%")
print(f"Our KNN (v1):          77.78%")
print(f"Random Forest (v2):    78.52%")
print(f"XGBoost (v3):          {acc*100:.2f}%")