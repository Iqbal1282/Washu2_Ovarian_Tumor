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
from multi_head_models2 import BinaryClassification
from dataset_washu2 import Classificaiton_Dataset

import subprocess
import re
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utils import plot_roc_curve
import numpy as np
import random 

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED) ; random.seed(SEED); 

max_epochs = 100
min_epochs = 1
batch_size = 16
check_val_every_n_epoch = 3
num_workers = 0
k_fold = 5

try: 
    # Get the latest Git commit message
    commit_string = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).decode("utf-8").strip()
    commit_string = re.sub(r'\W+', '_', commit_string) 
    commit_log = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

    
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    

all_fprs = []
all_tprs = []
all_aucs = []

project_title = "Ovarian Cancer Classification 8"
Experiment_Group = f"Exp4:{commit_string}_{commit_log}"
train_config = {
        "k_fold": k_fold,
        "batch_size": batch_size,
        "radiomics": False,
        "encoder_checkpoint": "normtverskyloss_binary_segmentation",
        "input_dim": 64,
        "loss_fn": "different loss functions experiements, where score is 2 score2 is 0.5 weight",
        "model_type": "BinaryClassification",
        "info": "Foun encoder median Score experiment",
        "info2": "patient greater than 120 are considered in testset"
    }

for fold in range(k_fold):
    # Initialize WandB Logger
    run_name = f'Fold_{fold}' #{commit_log}"_"{commit_string}"_{datetime.now()}'
    # wandb_logger = WandbLogger(
    #         log_model=False, project=project_title, name=run_name, 
    #     )
    
    wandb_logger = WandbLogger(
        log_model=False,
        project=project_title,
        name=run_name,
        group= Experiment_Group,
        tags=[f"fold_{fold}", "radiomics=False", f"commit_{commit_log}", "On Reviewed Cleaned Data", commit_string]
    )

    wandb_logger.experiment.config.update(train_config)


    trainDataset = Classificaiton_Dataset(phase = 'train', k_fold = k_fold, fold = fold, radiomics_dir= False) # r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
    valDataset = Classificaiton_Dataset(phase = 'val', k_fold = k_fold, fold = fold, radiomics_dir= False) #r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
    #testDataset =  Classificaiton_Dataset_test(phase = 'test', k_fold = k_fold, fold = fold, radiomics_dir= r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
    #testDataset = Classificaiton_Dataset(phase = 'val', k_fold = k_fold, fold = 0, radiomics_dir= False) #r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")
    testDataset = Classificaiton_Dataset(phase = 'test', radiomics_dir= False)

    train_loader = DataLoader(
                trainDataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last= True,
                #persistent_workers=True,
            )
    
    val_loader = DataLoader(
                valDataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last= True,
                #persistent_workers=True,
            )
    
    test_loader = DataLoader(
                testDataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last= False,
                #persistent_workers=True,
            )


    # Initialize Callbacks
    early_stopping = EarlyStopping(monitor="validation/loss", patience=100, mode="min")
    checkpoint_callback = ModelCheckpoint(
            monitor="validation/combined_score",
            mode="max",
            dirpath=f"checkpoints/washu2_classification/{commit_log}/{fold}",
            filename="best-checkpoint-{epoch:02d}-{validation/auc:.4f}",
            save_top_k=1,
        )


    # Initialize Trainer
    trainer = pl.Trainer(
            logger=wandb_logger,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=check_val_every_n_epoch,
            accelerator="gpu",
        )


    model = BinaryClassification(input_dim= 64, num_classes= 1,  encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt", radiomics= False)
    mmotu_checkpoint = torch.load(r"checkpoints\mmotu_classfication\auc=0.8749.ckpt", map_location="cpu")

    # Step 3: Filter out incompatible layers
    model_state = model.state_dict()
    filtered_checkpoint = {
        k: v for k, v in mmotu_checkpoint.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    # Step 4: Load filtered state_dict
    missing, unexpected = model.load_state_dict(filtered_checkpoint, strict=False)

    #missing, unexpected = model.load_state_dict(mmotu_checkpoint["state_dict"], strict=False)

    # Optionally, log missing/unexpected keys
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    
    trainer.fit(model, train_loader, val_loader)


    # Get predictions on the test set for ROC curve
    # Get predictions on the test set
    y_true, y_probs = model.get_predictions_on_loader(val_loader)
    #Load best model from checkpoint after training
    best_model_path = checkpoint_callback.best_model_path
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
    fpr, tpr, roc_auc = plot_roc_curve(y_true, y_probs, fold_idx=fold + 1, wandb_logger=wandb_logger)

    # Save ROC data for multi-fold plot
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    all_aucs.append(roc_auc)

    # Finish WandB Run
    wandb_logger.experiment.unwatch(model)
    wandb.finish()


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

final_img_path = 'plots/roc_all_folds.png'
plt.savefig(final_img_path)
plt.close()

# # Log final summary ROC to WandB
run_name =  f"All Fold AUC: {commit_log}"  # f'classification_all_folds_{commit_log}_commit_{commit_string}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
wandb_logger = WandbLogger(
    log_model=False,
    project=project_title,
    name=run_name,
    group=Experiment_Group,
    tags=[f"fold_all", "radiomics=False", f"commit_{commit_log}"]
)
wandb_logger.experiment.log({"ROC Curve - All Folds": wandb.Image(final_img_path)})
wandb.finish()