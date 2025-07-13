import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F 
import torch 
import torchvision.utils as vutils
import wandb
import torch
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchvision.models import resnet18 , ResNet18_Weights
import torchvision 


from losses import * 


import torch.nn as nn
import segmentation_models_pytorch as smp


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, self.eps, 1 - self.eps)

        if self.clip is not None and self.clip > 0:
            inputs_sigmoid = (inputs_sigmoid - self.clip).clamp(min=0, max=1)

        targets = targets.float()
        loss_pos = targets * torch.log(inputs_sigmoid) * (1 - inputs_sigmoid) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(1 - inputs_sigmoid) * inputs_sigmoid ** self.gamma_neg
        loss = -loss_pos - loss_neg
        return loss.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
    



class SDFModel(nn.Module):
    def __init__(self):
        super(SDFModel, self).__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.backbone(x)        # Output shape: (B, 1, H, W)
        x = self.activation(x)      # Output in [-1, 1]
        return x
    

class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 8, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(4)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        enc = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        enc = self.pool2(self.relu2(self.bn2(self.conv2(enc))))
        enc = self.pool3(self.relu3(self.bn3(self.conv3(enc))))
        enc = self.relu4(self.bn4(self.conv4(enc)))
        enc = self.relu5(self.bn5(self.conv5(enc)))
        enc = self.conv6(enc)
        return enc 
    
    
class MyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder using ConvTranspose2d
        self.deconv0 = nn.ConvTranspose2d(4, 128, kernel_size=4, stride=2, padding=1)  # Upsample to 8x8
        self.dbn0 = nn.BatchNorm2d(128)
        self.drelu0 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 16x16
        self.dbn1 = nn.BatchNorm2d(64)
        self.drelu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample to 32x32
        self.dbn2 = nn.BatchNorm2d(32)
        self.drelu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Upsample to 64x64
        self.dbn3 = nn.BatchNorm2d(16)
        self.drelu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # Upsample to 128x128
        self.dbn4 = nn.BatchNorm2d(8)
        self.drelu4 = nn.ReLU()

        # for the tumor head 
        self.deconv5 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)  # Upsample to 256x256


        # for the mask head 
        self.deconv6 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)  # Upsample to 256x256
        self.deconv7 = nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, padding=0)  # Upsample to 256x256

    def forward(self, enc):
        dec = self.drelu0(self.dbn0(self.deconv0(enc)))
        dec = self.drelu1(self.dbn1(self.deconv1(dec)))
        dec = self.drelu2(self.dbn2(self.deconv2(dec)))
        dec = self.drelu3(self.dbn3(self.deconv3(dec)))
        dec = self.drelu4(self.dbn4(self.deconv4(dec)))
        dec1 = self.deconv5(dec)  # Final layer, no activation to retain pixel values # tumor head 
        dec2 = self.deconv6(dec)     # mask head 
        dec2 = self.deconv7(dec2)  

        return dec1, dec2 

class Compress_Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = MyEncoder()
        self.decode = MyDecoder()
    def forward(self, x):
        x = self.encode(x)
        x1, x2 = self.decode(x)
        return x1, x2, x
        

# Define the CNN Model using PyTorch Lightning
class Compress_Segmentor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encode = MyEncoder()
        self.decode = MyDecoder()
        self.hybrid_loss = HybridLoss()
        self.tverskyloss = TverskyLoss()

    def forward(self, x):
        x = self.encode(x)
        x1, x2 = self.decode(x)
        return x1, x2

    def training_step(self, batch, batch_idx):
        x, y_temp = batch
        y_tumor, y_mask = self(x)
        y = x*y_temp
        loss =  self.tverskyloss(y_mask, y_temp)  + torch.log(1+ F.mse_loss(y_tumor, y))
        #acc = (y_hat.argmax(dim=1) == y).float().mean()
        #psnr = 
        self.log("train/loss", loss, prog_bar=True)
        #self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_temp = batch
        y_tumor, y_mask = self(x)
        y = x*y_temp
        loss =  self.tverskyloss(y_mask, y_temp)  + torch.log(1+ F.mse_loss(y_tumor, y))
        #loss = torch.log(1+F.mse_loss(y_hat, y))
        #acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("validation/loss", loss, prog_bar=True)
        #self.log("val_acc", acc, prog_bar=True)

         # Log images (only log a few samples)
        if batch_idx == 0:
            if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(y[:8], normalize=True, scale_each=True)
                grid_pred = vutils.make_grid(y_tumor[:8], normalize=True, scale_each=True)

                self.logger.experiment.log({
                    "val/input_images": wandb.Image(grid_input, caption="Input Images"),
                    "val/predicted_images": wandb.Image(grid_pred, caption="Predicted Images"),
                    "global_step": self.trainer.global_step
                })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

    
class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(FCNetwork, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class ImageNet_Model(nn.Module):
    def __init__(self, name = "resnet18", outsize = 1):

        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, outsize)

    def forward(self, x):
        return self.model(x)

class BinaryClassificationTorch(nn.Module):
    def __init__(self, input_dim=64, output_size = 5, num_classes=1, radiomics=False, radiomics_dim=463,
                 encoder_weight_path=None, sdf_model_path=None):
        super().__init__()

        self.input_size = input_dim
        self.hidden_sizes = [512, 128, 64, 32]
        self.hidden_sizes2 = [64, 32]
        self.output_size = output_size 
        self.radiomics = radiomics

        segmodel = Compress_Segmentor.load_from_checkpoint(encoder_weight_path, strict=True)
        encoder = segmodel.encode
        encoder.to("cpu")
        for p in encoder.parameters(): p.requires_grad = False
        self.encoder = encoder

        self.sdf_model = SDFModel()
        self.sdf_model.load_state_dict(torch.load(sdf_model_path))
        for p in self.sdf_model.parameters(): p.requires_grad = False

        self.full_encoder = ImageNet_Model(outsize=self.output_size)
        self.boundary_encoder = ImageNet_Model(outsize=self.output_size)
        self.center_encoder = ImageNet_Model(outsize=self.output_size)

        if radiomics:
            self.linear_radiomics = FCNetwork(radiomics_dim, [128, 64, 64], 32)
            self.linear_radiomics_tail = FCNetwork(32, [32, 32, 16], self.output_size)
            self.linear = FCNetwork(self.input_size * 2 + 32, self.hidden_sizes, self.output_size)
            self.linear_trainable = FCNetwork(self.input_size, self.hidden_sizes2, self.output_size)
        else:
            self.linear = FCNetwork(self.input_size, self.hidden_sizes, self.output_size)

        self.final_layer = FCNetwork(4 * self.output_size, [24, 12, 5], num_classes)

        self.loss_fn = FocalLoss()
        self.loss_fn2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))

    def normalize_sdf(self, sdf_image):
        sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        return sdf_image * 2 - 1

    def forward(self, x, x2_radiomics=None):
        x1 = self.encoder(x)
        x2 = self.full_encoder(x)

        x_sdf = self.sdf_model(x)
        x_sdf = self.normalize_sdf(x_sdf)

        lower_thresh = torch.empty(1).uniform_(-0.45, -0.35).item()
        upper_thresh = torch.empty(1).uniform_(0.35, 0.45).item()
        center_thresh = torch.empty(1).uniform_(0.3, 0.4).item()

        boundary_mask = (x_sdf < upper_thresh) & (x_sdf > lower_thresh)
        center_mask = (x_sdf < center_thresh)

        x3 = self.boundary_encoder(x * boundary_mask)
        x4 = self.center_encoder(x * center_mask)

        if x2_radiomics is not None:
            x2_radiomics = self.linear_radiomics(x2_radiomics)
            x_combined = torch.cat((x, x2_radiomics), dim=1)
            score_main = self.linear(x_combined).squeeze()
            score_tail_1 = self.linear_trainable(x2.reshape(x.shape[0], -1)).squeeze()
            score_tail_2 = self.linear_radiomics_tail(x2_radiomics).squeeze()
            return score_main, (score_tail_1, score_tail_2)
        else:
            x1 = self.linear(x1.reshape(x.shape[0], -1))
            x = torch.cat([x1, x2, x3, x4], dim=-1)
            return self.final_layer(x).squeeze(), (x1.squeeze(), x2.squeeze(), x3.squeeze(), x4.squeeze())

    def compute_loss(self, x, y, x2_rad=None):
        if x2_rad is not None:
            score, tails = self.forward(x, x2_rad)
            loss = self.loss_fn(score, y.float()) + sum(self.loss_fn(t, y.float()) for t in tails)
        else:
            score, tails = self.forward(x)
            y2 = y.unsqueeze(-1).repeat((1, self.output_size)).squeeze()
            loss = (self.loss_fn(score, y.float()) * 0.2 +
                    self.loss_fn(tails[0], y2.float()) * 0.1 +
                    sum(self.loss_fn(t, y2.float()) for t in tails[1:]) +
                    self.loss_fn2(score, y.float()) * 0.2 +
                    self.loss_fn2(tails[0], y2.float()) * 0.1 +
                    sum(self.loss_fn2(t, y2.float()) for t in tails[1:]))
        return loss

    def predict_on_loader(self, dataloader):
        self.eval()
        all_probs, all_targets = [], []

        device = next(self.parameters()).device  # Automatically detect model's device

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    scores, tails = self.forward(x)

                    
                else:
                    x, x2, y = batch
                    x, x2, y = x.to(device), x2.to(device), y.to(device)
                    scores, tails = self.forward(x, x2)

                scores = scores*0.4
                for  s in tails: 
                    scores += s.mean(dim = -1)*0.15

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

                    
                probs = torch.sigmoid(scores)
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())

        return torch.cat(all_targets).numpy(), torch.cat(all_probs).numpy()
    

if __name__ == "__main__":
    model = BinaryClassificationTorch(input_dim= 64, num_classes= 1,  
                                 encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\a56e77a\best-checkpoint-epoch=77-validation\loss=0.2544.ckpt", 
                                 sdf_model_path= r"checkpoints\deeplabv3_sdf_randomcrop\model_20250711_201243\epoch_84",
                                 radiomics= False)
    print(model)
    model.eval()
    print(model(torch.randn(1, 1,256, 256))) #, torch.randn(1, 1,256, 256)).shape)
