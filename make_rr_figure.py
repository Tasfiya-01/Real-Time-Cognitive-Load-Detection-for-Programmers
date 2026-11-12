import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv('results/rr_summary.csv', dtype={'pid': str})
bars = [('stew_auc', 'STEW model (no retraining)', '#9ca3af'),
        ('frontal_auc', 'Programmer model: frontal only', '#fcd34d'),
        ('posterior_auc', 'Programmer model: posterior only', '#60a5fa'),
        ('all_auc', 'Programmer model: all 14 channels', '#1d4ed8')]

x = np.arange(len(df)); wdt = 0.2
fig, ax = plt.subplots(figsize=(12, 4.5))
for k, (col, label, color) in enumerate(bars):
    ax.bar(x + (k - 1.5) * wdt, df[col], wdt, color=color,
           label=f'{label} (mean {df[col].mean():.2f})')
ax.axhline(0.5, color='black', lw=1, ls='--')
ax.text(-0.55, 0.51, 'chance', ha='left', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels('P' + df.pid)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel('AUC: code reading vs rest (new participant)')
ax.set_title('Leave-one-subject-out: programmer-trained model detects reading vs rest in every participant')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
plt.savefig('results/fig_reading_vs_rest.png', dpi=200, bbox_inches='tight')
print('saved results/fig_reading_vs_rest.png')