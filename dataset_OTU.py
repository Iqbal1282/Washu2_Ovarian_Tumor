import os 
import pandas as pd 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset    

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    A.CenterCrop(256, 256, p=0.5),
    #A.RandomCrop(256, 256, p =0.9),
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.Normalize(mean=(0.5,), std=(0.5,)),  # Adjust if using RGB
    ToTensorV2()
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])


class SegmentImageDataset(Dataset):
    def __init__(
        self,
        root_dir="data/OTU_2d",
        phase="train",
        test_mode=False,
    ):
        self.root_dir = root_dir
        self.phase = phase
        self.test_mode = test_mode

        # Load image filenames
        unique_img_files = os.listdir(os.path.join(self.root_dir, "images"))
        num_lines = len(unique_img_files)
        separation_index = int(0.85 * num_lines)

        if self.phase == "train":
            self.image_files = unique_img_files[:separation_index]
            self.transform = train_transform
        else:
            self.image_files = unique_img_files[separation_index:]
            self.transform = val_transform

        # Store image-mask pairs
        self.data = [
            (os.path.join(self.root_dir, "images", file),
             os.path.join(self.root_dir, "masks", file.split(".")[0] + "_binary.PNG"))
            for file in self.image_files
        ]

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        # Load image and mask as grayscale arrays
        image = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        # Optional: Binarize mask if needed
        mask = (mask > 0).astype(np.float32)

        # Apply transforms (Albumentations expects HxW or HxWxC, not CxHxW)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)  # Add channel dimension for mask
        else:
            # Convert to tensors and add channel dim manually
            image = torch.tensor(image).unsqueeze(0)
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask
    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    # Example usage
    import matplotlib.pyplot as plt
    dataset = SegmentImageDataset(phase='train')
    print("Dataset size: ", len(dataset))
    fig, ax = plt.subplots(10, 2, figsize=(5, 20))
    for i in range(10):
        img, mask = dataset[i]
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        print(f"Image max value: {img.max()}, Mask max value: {mask.max()}")
        ax[i, 0].imshow(img.squeeze(0), cmap='gray')
        ax[i, 1].imshow(mask.squeeze(0), cmap='gray')
        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Mask")  
    plt.tight_layout()
    plt.show()

