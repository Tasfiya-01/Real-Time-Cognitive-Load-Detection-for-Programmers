import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

print("Loading data and model...")
X = np.load('data/X_features_v3.npy')
y = np.load('data/y_labels.npy')
model  = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler_v3.pkl')

X_scaled = scaler.transform(X)

# Feature names
channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
feature_types = ['Delta','Theta','Alpha','Beta','Gamma',
                 'Theta/Alpha','Beta/Alpha','Theta/Beta',
                 'Mean','Std','Var','Skew','Kurt','RMS',
                 'SampEn','PermEn','SpecEn','SVDEn']
feature_names = [f"{ch}_{ft}" for ch in channels for ft in feature_types]

print("Computing SHAP values (2-3 minutes)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled[:200])

print("Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: Top 20 features
shap_abs = np.abs(shap_values).mean(axis=(0, 2)) if len(np.array(shap_values).shape) == 3 else np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(shap_abs)[-20:][::-1]
top_names = [feature_names[i] for i in top_idx]
top_vals  = shap_abs[top_idx]

axes[0].barh(range(20), top_vals[::-1], color='steelblue')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(top_names[::-1], fontsize=9)
axes[0].set_title('Top 20 Important Features (SHAP)', fontsize=13)
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].grid(True, alpha=0.3)

# Plot 2: Channel importance
channel_importance = {}
for i, name in enumerate(feature_names):
    ch = name.split('_')[0]
    channel_importance[ch] = channel_importance.get(ch, 0) + shap_abs[i]

ch_names = list(channel_importance.keys())
ch_vals  = list(channel_importance.values())
colors = ['red' if v == max(ch_vals) else 'steelblue' for v in ch_vals]

axes[1].bar(ch_names, ch_vals, color=colors)
axes[1].set_title('EEG Channel Importance (SHAP)', fontsize=13)
axes[1].set_xlabel('Channel')
axes[1].set_ylabel('Total SHAP Importance')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/shap_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved! results/shap_analysis.png")

print("\nTop 5 Most Important Features:")
for i, (name, val) in enumerate(zip(top_names[:5], top_vals[:5])):
    print(f"  {i+1}. {name}: {val:.4f}")

print("\nTop 3 Most Important Channels:")
sorted_ch = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
for i, (ch, val) in enumerate(sorted_ch[:3]):
    print(f"  {i+1}. {ch}: {val:.4f}")