import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch 
import pandas as pd 
from pathlib import Path
import nrrd
import torch
from torch.utils.data import Dataset
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
from PIL import Image
from sklearn.model_selection import KFold
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


# # Training transformations
# common_train_transform = A.Compose([
# 	A.PadIfNeeded(min_height=284, min_width=284, border_mode=0, value=0),  
# 	A.Resize(256, 256),  
# 	A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit= 20, p =0.5),
# 	A.CenterCrop(256, 256, p=0.5),
# 	#A.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [0,1] range
# 	#ToTensorV2(),
# ])

# # Validation transformations
# common_val_transform = A.Compose([
# 	A.PadIfNeeded(min_height=284, min_width=284, border_mode=0, value=0),
# 	A.Resize(256, 256),
# 	#A.Normalize(mean=[0.5], std=[0.5]),
# 	#ToTensorV2(),
# ])

# # Normalization for images
# transform_img = A.Compose([
# 	#A.Normalize(mean=[58.42], std=[51.01]),
# 	ToTensorV2(),
# ])

# # Transform mask: Convert to binary mask and tensor
# transform_mask = A.Compose([
# 	#A.Lambda(image=lambda x: (x > 0).float()),  
# 	ToTensorV2(),
# ])


import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    #A.RandomCrop(256, 256),
	# A.RandomResizedCrop(scale=(0.95, 1.0),
	# 					ratio=(0.95, 1.05),
	# 					size=(256, 256)), 
	A.Resize(256, 256),
	

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

	A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=10, border_mode=0, value=0, p=0.5), 
    #A.ElasticTransform(alpha = 1, sigma = 50, p=0.8),
    A.GridDistortion(distort_limit=(-0.1,0.1), p=0.5),
	# #A.GaussNoise(var_limit=(0.2, 0.44), p=1),
    A.RandomBrightnessContrast(brightness_limit=(0, 0.01), contrast_limit=(0, 0.01), p=0.5),
    # #A.CLAHE(clip_limit=.5, tile_grid_size=(8, 8), p=0.5),
    # A.Downscale(scale_range=(0.8,0.99), p=0.5),
	

    A.Normalize(mean=(0.5,), std=(0.5,)),  # Adjust if using RGB
    ToTensorV2()
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0, mask_value=0),
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])



 # This dataset loader will be used for experiment with single image based (MR1 or MR2) model , where model is encoder(narrow) + fc regression model   
class Classificaiton_Dataset(Dataset):
	def __init__(
				self,
				root_dir=r"data/Ovarian_Reviewed_Data",
				response_dir=r"data/PAT_imaging_record.xlsx",
				radiomics_dir=False,
				phase="train",  # "train", "val", or "test"
				k_fold=5,
				fold=0
			):
		self.root_dir = root_dir
		self.phase = phase
		self.radiomics_dir = radiomics_dir

		# === Load radiomics features ===
		if radiomics_dir:
			radiomics_db = pd.read_csv(radiomics_dir)
			print("Radiomics features shape: ", radiomics_db.shape)

			radiomics_db['ImagePath'] = radiomics_db['ImagePath'].apply(
				lambda p: os.path.normpath(p).replace("\\", "/").split("Images/")[-1].replace("Patient_", "p")
			)
			radiomics_db = radiomics_db.set_index("ImagePath")
			radiomics_db = radiomics_db.drop(columns=["PatientID", "Side", "GT", "ImagePath", "Patient ID"], errors='ignore')

			scaler = StandardScaler()
			scaled_values = scaler.fit_transform(radiomics_db.values)
			radiomics_db.loc[:, :] = scaled_values

		# === Load response table and generate GT map ===
		df = pd.read_excel(response_dir, sheet_name="ROI STATS V4 (3)")
		df = df.dropna(subset=["Patient ID", "Side", "GT"])
		df["Patient ID"] = df["Patient ID"].astype(int)
		df["Side"] = df["Side"].astype(str).str.strip()
		df["GT"] = pd.to_numeric(df["GT"], errors="coerce").astype("Int64")
		df = df.dropna(subset=["GT"])
		df["GT"] = (df["GT"].astype(int)<1).astype(int)

		df["PatientSide"] = df.apply(lambda row: f"p{row['Patient ID']}_{row['Side']}", axis=1)

		# === Define test set based on Patient ID < 21 ===
		df["IsTest"] = df["Patient ID"] >= 120
		test_case_set = set(df[df["IsTest"]]["PatientSide"].tolist())

		# === Construct GT map ===
		grouped_gt = df.groupby("PatientSide")["GT"].agg(lambda x: x.mode()[0])
		case_ids = grouped_gt.index.tolist()
		case_labels = grouped_gt.values.tolist()
		label_map = grouped_gt.to_dict()

		# === Select cases based on phase ===
		if self.phase == "test":
			selected_cases = test_case_set
			self.transform = val_transform
		else:
			# Exclude test cases from fold-based splitting
			strat_case_ids = [cid for cid in case_ids if cid not in test_case_set]
			strat_case_labels = [label_map[cid] for cid in strat_case_ids]

			skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
			splits = list(skf.split(strat_case_ids, strat_case_labels))
			train_idx, val_idx = splits[fold]

			if self.phase == "train":
				selected_cases = set(strat_case_ids[i] for i in train_idx)
				self.transform = train_transform
			else:
				selected_cases = set(strat_case_ids[i] for i in val_idx)
				self.transform = val_transform

		# === Load filesystem and match against selected cases ===
		all_samples = []

		for patient_id in os.listdir(root_dir):
			patient_path = os.path.join(root_dir, patient_id)
			if not os.path.isdir(patient_path):
				continue

			for side in os.listdir(patient_path):
				side_path = os.path.join(patient_path, side)
				if not os.path.isdir(side_path):
					continue

				case_key = f"{patient_id}_{side}"
				if case_key not in selected_cases or case_key not in label_map:
					continue

				label = label_map[case_key]

				for fname in os.listdir(side_path):
					full_path = os.path.join(side_path, fname)
					if not os.path.isfile(full_path):
						continue
					if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
						continue

					if radiomics_dir:
						relative_path = os.path.relpath(full_path, start=self.root_dir).replace("\\", "/")
						if relative_path in radiomics_db.index:
							radiomics = radiomics_db.loc[relative_path].values.astype(float)
							all_samples.append((full_path, label, radiomics))
						else:
							print(f"⚠️ Radiomics not found for {relative_path}")
							continue
					else:
						all_samples.append((full_path, label))

		self.data = all_samples
				
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		if self.radiomics_dir: 
			image_path, response, radiomics_features = self.data[index]
		else: 
			image_path, response= self.data[index]
		#print("image_path: ", image_path)
		image = np.array(Image.open(image_path).convert('L'))
		if self.transform:
			transformed = self.transform(image=image)
			if self.radiomics_dir:
				return transformed["image"], torch.tensor(radiomics_features, dtype = torch.float32), torch.tensor(response, dtype=torch.float32)
			else:
				return transformed["image"], torch.tensor(response, dtype=torch.float32)
				
		else:
			if self.radiomics_dir:
				return torch.from_numpy(image), torch.tensor(radiomics_features, dtype = torch.float32),  torch.tensor(response, dtype=torch.float32)
			else:
				return torch.from_numpy(image), torch.tensor(response, dtype=torch.float32)	
			
	
if __name__ == '__main__':
	#Classificaiton_Dataset(phase = 'train', img_transform= transform_img)
	train_dataset = Classificaiton_Dataset(phase = 'train', k_fold=10, fold= 2)
	print("train dataset size: ", len(train_dataset))
	#print("test dataset size: ", len(train_dataset))
	#print("data sample: ", train_dataset.data) 
	# print(train_dataset[1][0].shape)
	# print(train_dataset[1][1].shape)
	# #print(train_dataset[21][2])
	# print(train_dataset[1][0].max())
	# print(train_dataset[1][1].max())
	# print(len(train_dataset))

	for i in range(5):
		if train_dataset.radiomics_dir: 
			img, radiomics, label = train_dataset[i]
		else: 
			img, label = train_dataset[i]
		#print("radiomics: ", radiomics.shape)
		print(label)
		print("image max: ", img.max(), "image min: ", img.min())
		plt.figure()
		#plt.subplot(1,3,1)
		plt.imshow(img[0], cmap= 'gray')
		plt.show()


	Malignant_count = 0 
	Benign_count = 0 

	if train_dataset.radiomics_dir: 
		for i in range(len(train_dataset)):
			if train_dataset[i][2] == 0: 
				Benign_count+=1 
			else: 
				Malignant_count+=1
		print("response: ", Malignant_count)
		print("no response: ", Benign_count)
	else:
		for i in range(len(train_dataset)):
			if train_dataset[i][1] == 0: 
				Benign_count+=1 
			else: 
				Malignant_count+=1
		print("response: ", Malignant_count)
		print("no response: ", Benign_count)


	# ## Train set statistics: 
	# # Malignant:  61
	# # Benign:  252
	# ##  Test set statistics:
	# # Malignant:  16
	# # Benign:  71
	# # #  Total:  77 malignant, 323 benign	
	# #pass 


