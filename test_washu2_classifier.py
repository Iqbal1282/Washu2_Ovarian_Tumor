import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from datetime import datetime
from models import BinaryClassification
from dataset_washu2_p1_43 import Classificaiton_Dataset
import subprocess
import re
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import plot_roc_curve
import numpy as np


batch_size = 64
num_workers = 0

testDataset =  Classificaiton_Dataset(phase = 'test')
test_loader = DataLoader(
                testDataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last= False,
                #persistent_workers=True,
            )

checkpoints_paths = [
    r"checkpoints\washu2_classification\81bc005\0\best-checkpoint-epoch=92-validation\auc=0.8953.ckpt",
    r"checkpoints\washu2_classification\81bc005\1\best-checkpoint-epoch=47-validation\auc=0.9783.ckpt",
    r"checkpoints\washu2_classification\81bc005\2\best-checkpoint-epoch=08-validation\auc=0.8519.ckpt",
    r"checkpoints\washu2_classification\81bc005\3\best-checkpoint-epoch=17-validation\auc=0.9172.ckpt",
    r"checkpoints\washu2_classification\81bc005\4\best-checkpoint-epoch=11-validation\auc=0.6433.ckpt",
]



all_fprs = []
all_tprs = []
all_aucs = []

for fold, best_model_path in enumerate(checkpoints_paths):
    #best_model_path = checkpoint_callback.best_model_path
    best_model = BinaryClassification.load_from_checkpoint(
        best_model_path,
        input_dim=64,
        num_classes=1,
        encoder_weight_path=r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt"
    )
    best_model.eval()
    best_model.freeze()

    # Get predictions using the best model
    y_true, y_probs = best_model.get_predictions_on_loader(test_loader)

    # Plot and log the ROC for this fold
    fpr, tpr, roc_auc = plot_roc_curve(y_true, y_probs, fold_idx=fold + 1, wandb_logger=None)

    # Save ROC data for multi-fold plot
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    all_aucs.append(roc_auc)
######################################### MULTI-FOLD ROC CURVE PLOTTING #########################################

# Create common FPR base for interpolation
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

final_img_path = 'roc_all_folds.png'
plt.savefig(final_img_path)
plt.close()
