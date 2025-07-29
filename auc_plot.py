import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Load saved ROC data
with open('plots/roc_data.pkl', 'rb') as f:
    roc_data = pickle.load(f)

all_fprs = roc_data['all_fprs']
all_tprs = roc_data['all_tprs']
mean_fpr = roc_data['mean_fpr']
mean_tpr = roc_data['mean_tpr']
mean_auc = roc_data['mean_auc']

# Interpolate all TPRs to the common mean_fpr grid
interp_tprs = []
for fpr, tpr in zip(all_fprs, all_tprs):
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0  # Force zero at beginning
    interp_tprs.append(interp_tpr)

interp_tprs = np.array(interp_tprs)
mean_tpr = np.mean(interp_tprs, axis=0)
std_tpr = np.std(interp_tprs, axis=0)

# Recompute mean AUC
mean_auc = auc(mean_fpr, mean_tpr)

# --- Plot ROC Cloud with Shaded Region ---
plt.figure(figsize=(8, 6))

# # Plot individual ROC curves lightly
# for tpr in interp_tprs:
#     plt.plot(mean_fpr, tpr, color='gray', lw=1, alpha=0.2)

# Plot mean ROC curve
plt.plot(mean_fpr, mean_tpr, color='blue', lw=2.5, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')

# Shaded region: mean ± std
plt.fill_between(mean_fpr,
                 np.maximum(mean_tpr - std_tpr, 0),
                 np.minimum(mean_tpr + std_tpr, 1),
                 color='blue',
                 alpha=0.2,
) #label='±1 Std. Dev.')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', lw=1)

# Aesthetics
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

# Save and show
plt.tight_layout()
plt.savefig('plots/roc_cloud_with_std_shade.png')
plt.show()
