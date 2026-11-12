import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load model and data
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

X = np.load('data/X_features_v2.npy')
y = np.load('data/y_labels.npy')

X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 1. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Low', 'Medium', 'High']
)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved!")

# 2. Class Accuracy Bar Chart
class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    ['Low', 'Medium', 'High'],
    class_acc * 100,
    color=['#00e676', '#ffb300', '#ff1744'],
    width=0.5
)
for bar, val in zip(bars, class_acc):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{val*100:.1f}%',
        ha='center', fontsize=12, fontweight='bold'
    )
ax.set_ylim(0, 110)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Accuracy per Class', fontsize=16, fontweight='bold')
ax.set_xlabel('Cognitive Load Level', fontsize=13)
plt.tight_layout()
plt.savefig('results/class_accuracy.png', dpi=150)
plt.close()
print("Class accuracy saved!")

# 3. Feature Importance
importances = np.abs(np.corrcoef(X_scaled.T, y)[:-1, -1])
top_idx = np.argsort(importances)[-15:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    [f'Feature {i}' for i in top_idx],
    importances[top_idx],
    color='#7c6ee0'
)
ax.set_xlabel('Correlation with Load Level', fontsize=13)
ax.set_title('Top 15 Important Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
plt.close()
print("Feature importance saved!")

print("\nAll graphs saved in results/ folder!")