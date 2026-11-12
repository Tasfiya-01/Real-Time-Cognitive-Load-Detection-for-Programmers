import pandas as pd, numpy as np, matplotlib.pyplot as plt, ast

s = pd.read_csv('results/summary_participants.csv', dtype={'pid': str})
fig, ax = plt.subplots(1, 2, figsize=(13, 4))

# Figure 1: per-participant correlation with code complexity
ax[0].bar('P' + s.pid, s.rho_cc_max, color=['#3b82f6' if r > 0 else '#ef4444' for r in s.rho_cc_max])
ax[0].axhline(0, color='black', lw=0.8)
ax[0].set_ylabel('Spearman rho (load score vs cyclomatic complexity)')
ax[0].set_title('STEW model: no consistent link to code difficulty')

# Figure 2: share of windows predicted low / medium / high per participant
c = np.array([ast.literal_eval(x) for x in s.pred_low_med_high], dtype=float)
c = c / c.sum(1, keepdims=True)
bottom = np.zeros(len(s))
for i, (lab, col) in enumerate(zip(['low', 'medium', 'high'], ['#93c5fd', '#fcd34d', '#f87171'])):
    ax[1].bar('P' + s.pid, c[:, i], bottom=bottom, label=lab, color=col)
    bottom += c[:, i]
ax[1].set_ylabel('share of 5-s windows')
ax[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)   # legend outside the plot
ax[1].set_title('Predictions depend mostly on the person')

plt.tight_layout()
plt.savefig('results/fig_stew_transfer.png', dpi=200, bbox_inches='tight')
print('saved results/fig_stew_transfer.png')