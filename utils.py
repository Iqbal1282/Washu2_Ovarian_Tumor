from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import wandb

def plot_roc_curve(y_true, y_probs, fold_idx=None, wandb_logger=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold_idx} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx}')
    plt.legend(loc='lower right')
    plt.grid(True)

    img_path = f'plots/roc_curve_fold_{fold_idx}.png'
    plt.savefig(img_path)
    plt.close()

    if wandb_logger:
        wandb_logger.experiment.log({f'ROC Curve Fold {fold_idx}': wandb.Image(img_path)})

    return fpr, tpr, roc_auc
