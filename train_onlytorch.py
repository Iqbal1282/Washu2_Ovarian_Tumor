import torch
import wandb
import numpy as np
import random
import re
import subprocess
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from models_onlytorch import BinaryClassificationTorch
from dataset_washu2 import Classificaiton_Dataset
from utils import plot_roc_curve, compute_weighted_accuracy
from tqdm import tqdm 
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
# Set deterministic behavior
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

# Settings
max_epochs = 100
batch_size = 16
k_fold = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Git Commit Info
try:
    commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
    commit_string = re.sub(r'\W+', '_', commit_string)
    commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
except Exception as e:
    commit_string, commit_log = "no_commit", "0000"
    print(f"Git commit fetch failed: {e}")

# WandB Settings
project_title = "Torch Ovarian Cancer Classification 10"
experiment_group = f"Exp4:{commit_string}_{commit_log}"
train_config = {
    "k_fold": k_fold,
    "batch_size": batch_size,
    "radiomics": False,
    "encoder_checkpoint": "normtverskyloss_binary_segmentation",
    "input_dim": '256x256:64',
    "model_type": "BinaryClassificationTorch",
    "info": "Training without PyTorch Lightning",
}

# Storage for fold-wise metrics
all_fprs, all_tprs, all_aucs = [], [], []

# --- Start K-Fold Training ---
for fold in range(k_fold):
    run = wandb.init(project=project_title, name=f"Fold_{fold}", group=experiment_group, config=train_config)
    
    # Datasets & Loaders
    train_dataset = Classificaiton_Dataset(phase='train', k_fold=k_fold, fold=fold, radiomics_dir=False)
    val_dataset = Classificaiton_Dataset(phase='val', k_fold=k_fold, fold=fold, radiomics_dir=False)
    test_dataset = Classificaiton_Dataset(phase='test', radiomics_dir=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = BinaryClassificationTorch(input_dim= 64, num_classes= 1,  
                                 encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt", 
                                 sdf_model_path= r"checkpoints\deeplabv3_sdf_randomcrop\model_20250711_201243\epoch_84",
                                 radiomics= False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    best_val_auc = -1
    best_combined_score = -1 
    best_model_state = None 
    
    # Metrics
    accuracy_metric = BinaryAccuracy().to(device)
    auc_metric = BinaryAUROC().to(device)

    # --- Training Loop ---
    for epoch in tqdm(range(max_epochs), leave= False):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 2:
                x, y = batch
                loss = model.compute_loss(x.to(device), y.to(device))
            else:
                x, x2, y = batch
                loss = model.compute_loss(x.to(device), y.to(device), x2.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({f"train/loss_fold_{fold}": avg_train_loss, "epoch": epoch})

        # --- Validation Evaluation ---
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores, tails = model(x)
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores, tails = model(x, x2)

                # # Ensure main scores are 1D
                # scores = scores.view(-1)

                # # Stack all tails after computing per-sample means
                # tail_means = [tail.mean(dim=-1) for tail in tails]  # Each: (batch_size,)
                # tail_means_stacked = torch.stack(tail_means, dim=0)  # Shape: (num_tails, batch_size)

                # # Compute median across tails: shape (batch_size,)
                # tail_median = torch.median(tail_means_stacked, dim=0).values

                # # Average main score and tail median
                # final_score = 0.5 * scores + 0.5 * tail_median

                # scores = final_score

                scores = scores*0.4
                for  s in tails: 
                    scores += s.mean(dim = -1)*0.15

                probs = torch.sigmoid(scores)
                y_probs.append(probs)
                y_true.append(y)

                accuracy_metric.update(probs, y.int())
                auc_metric.update(probs, y.int())

        y_true = torch.cat(y_true)
        y_probs = torch.cat(y_probs)

        # Compute metrics
        val_accuracy = accuracy_metric.compute().item()
        val_auc = auc_metric.compute().item()
        val_wacc = compute_weighted_accuracy(y_probs, y_true)

        # Reset metrics
        accuracy_metric.reset()
        auc_metric.reset()

        # ROC
        fpr, tpr, roc_auc = plot_roc_curve(y_true.cpu().numpy(), y_probs.cpu().numpy(), fold_idx=fold + 1)

        combined_score = 0.2*val_wacc + 0.3* val_accuracy + 0.5* roc_auc 

        # Log all metrics
        wandb.log({
            f"val/roc_auc_fold_{fold}": val_auc,
            f"val/accuracy_fold_{fold}": val_accuracy,
            f"val/weighted_accuracy_fold_{fold}": val_wacc,
            #f"val/roc_curve_fold_{fold}": wandb.Image(f"plots/roc_curve_fold_{fold+1}.png"),
            "epoch": epoch
        })

        if roc_auc > best_val_auc:
            best_val_auc = roc_auc
            best_model_state = model.state_dict()

        # if combined_score > best_combined_score:
        #     best_combined_score = roc_auc
        #     best_model_state = model.state_dict()

    # --- Load Best Model and Test ---
    model.load_state_dict(best_model_state)
    y_true, y_probs = model.predict_on_loader(test_loader)
    fpr, tpr, roc_auc = plot_roc_curve(y_true, y_probs, fold_idx=fold + 1)
    wandb.log({
        #f"test/roc_auc_fold_{fold}": roc_auc,
        f"test/roc_curve_fold_{fold}": wandb.Image(f"plots/roc_curve_fold_{fold+1}.png"),
    })
    
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    all_aucs.append(roc_auc)
    run.finish()

# --- Plot Multi-Fold ROC Curve ---
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

plt.figure()
for i, (fpr, tpr, auc_score) in enumerate(zip(all_fprs, all_tprs, all_aucs)):
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i+1} (AUC = {auc_score:.2f})')

mean_tpr = np.mean(interp_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='b', lw=2, linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Across All Folds')
plt.legend(loc='lower right')
plt.grid(True)
os.makedirs("plots", exist_ok=True)
final_img_path = 'plots/roc_all_folds.png'
plt.savefig(final_img_path)
plt.close()

# Final ROC to WandB
final_run = wandb.init(
    project=project_title,
    name=f"All_Folds_{commit_log}",
    group=experiment_group,
)
final_run.log({"ROC Curve - All Folds": wandb.Image(final_img_path)})
final_run.finish()
