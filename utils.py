from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import wandb
import os 

# def plot_roc_curve(y_true, y_probs, fold_idx=None, wandb_logger=None):
#     fpr, tpr, thresholds = roc_curve(y_true, y_probs)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, lw=2, label=f'Fold {fold_idx} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--', lw=1)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve - Fold {fold_idx}')
#     plt.legend(loc='lower right')
#     plt.grid(True)

#     img_path = f'plots/roc_curve_fold_{fold_idx}.png'
#     plt.savefig(img_path)
#     plt.close()

#     if wandb_logger:
#         wandb_logger.experiment.log({f'ROC Curve Fold {fold_idx}': wandb.Image(img_path)})

#     return fpr, tpr, roc_auc


def plot_roc_curve(y_true, y_probs, fold_idx=None, wandb_logger=None):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Compute accuracy (default threshold 0.5)
    y_pred = (y_probs >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'Fold {fold_idx} (AUC = {roc_auc:.2f}, Acc = {accuracy:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold_idx}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save and log image
    os.makedirs("plots", exist_ok=True)
    img_path = f'plots/roc_curve_fold_{fold_idx}.png'
    plt.savefig(img_path)
    plt.close()

    if wandb_logger:
        wandb_logger.experiment.log({
            f'ROC Curve Fold {fold_idx}': wandb.Image(img_path),
            f'fold_{fold_idx}_auc': roc_auc,
            f'fold_{fold_idx}_accuracy': accuracy
        })

    return fpr, tpr, roc_auc
