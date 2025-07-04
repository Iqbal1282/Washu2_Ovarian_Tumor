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
	A.RandomResizedCrop(scale=(0.95, 1.0),
						ratio=(0.95, 1.05),
						size=(256, 256),
						),
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




 # This dataset loader will be used for experiment with single image based (MR1 or MR2) model , where model is encoder(narrow) + fc regression model   
class Classificaiton_Dataset(Dataset):
	def __init__(
		self,
		root_dir= r"data\washu_p1_43\US_ROI_whole_image_filtered_qz",
		response_dir=r"data\PAT_imaging_record.xlsx",
		radiomics_dir= r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf3_nd_normseg.csv",
		phase="test",
		k_fold=5,
		fold=0,
	):
		self.root_dir = root_dir
		self.phase = phase
		#self.img_transform = img_transform

		self.radiomics_dir = radiomics_dir
		if radiomics_dir:
			radiomics_db = pd.read_csv(radiomics_dir)
			print("Radiomics features shape: ", radiomics_db.shape)

			# Normalize paths to relative format
			radiomics_db['ImagePath'] = radiomics_db['ImagePath'].apply(
				lambda p: (os.path.normpath(p).replace("\\", "/").split("Images/")[-1]).replace("Patient_", "p")
			)

			radiomics_db = radiomics_db.set_index("ImagePath")
			radiomics_db = radiomics_db.drop(columns=["PatientID", "Side", "GT", "ImagePath", "Patient ID"], errors='ignore')

			# Normalize features
			scaler = StandardScaler()
			scaled_values = scaler.fit_transform(radiomics_db.values)
			radiomics_db.loc[:, :] = scaled_values
		# Step 1: Read Excel sheet
		df = pd.read_excel(response_dir, sheet_name="ROI STATS V3")
		df = df.dropna(subset=["Patient ID", "Side", "GT"])
		df["Patient ID"] = df["Patient ID"].astype(int).astype(str).str.strip()
		df["Side"] = df["Side"].astype(str).str.strip()
		df["GT"] = pd.to_numeric(df["GT"], errors="coerce").astype("Int64")
		df = df.dropna(subset=["GT"])
		df["GT"] = df["GT"].astype(int)

		label_map = {
			("p" + row["Patient ID"], row["Side"]): row["GT"]
			for _, row in df.iterrows()
		}

		
		# Step 2: Get all patient IDs and apply k-fold
		all_patient_ids = sorted([
			d for d in os.listdir(root_dir)
			if os.path.isdir(os.path.join(root_dir, d))
		])

		# kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
		# splits = list(kf.split(all_patient_ids))
		# train_idx, val_idx = splits[fold]

		selected_patient_ids = all_patient_ids
		self.transform = val_transform 

		# Step 3: Gather all samples
		all_samples = []
		for patient_id in selected_patient_ids:
			patient_path = os.path.join(root_dir, patient_id)
			for side in os.listdir(patient_path):
				side_path = os.path.join(patient_path, side)
				if not os.path.isdir(side_path):
					continue

				key = (patient_id, side)
				if key not in label_map:
					continue

				label = label_map[key]
				label = 1 if label == 0 else 0  # Optional: relabel if needed

				for fname in os.listdir(side_path):
					# Inside: for fname in os.listdir(side_path):
					full_path = os.path.join(side_path, fname)
					if os.path.exists(full_path) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
						if radiomics_dir:
							# Construct relative path like: p10/R/image.jpg
							relative_path = os.path.relpath(full_path, start=self.root_dir).replace("\\", "/")

							if relative_path in radiomics_db.index:
								radiomics = radiomics_db.loc[relative_path].values.astype(float)
								all_samples.append((full_path, label, radiomics))
							else:
								# Optional: try loose matching or log the missing ones
								print(f"⚠️ Radiomics not found for {relative_path}")
								continue
						else:
							all_samples.append((full_path, label))

		self.data = all_samples
			
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		image_path, response, radiomics_features = self.data[index]
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
	# #Classificaiton_Dataset(phase = 'train', img_transform= transform_img)
	# train_dataset = Classificaiton_Dataset(phase = 'train')
	# print("train dataset size: ", len(train_dataset))
	# #print("test dataset size: ", len(train_dataset))
	# #print("data sample: ", train_dataset.data) 
	# # print(train_dataset[1][0].shape)
	# # print(train_dataset[1][1].shape)
	# # #print(train_dataset[21][2])
	# # print(train_dataset[1][0].max())
	# # print(train_dataset[1][1].max())
	# # print(len(train_dataset))

	# for i in range(10):
	# 	img, radiomics, label = train_dataset[i]
	# 	print("radiomics: ", radiomics.shape)
	# 	print(label)
	# 	plt.figure()
	# 	#plt.subplot(1,3,1)
	# 	plt.imshow(img[0], cmap= 'gray')
	# 	plt.show()


	# Malignant_count = 0 
	# Benign_count = 0 

	# for i in range(len(train_dataset)):
	# 	if train_dataset[i][2] == 0: 
	# 		Benign_count+=1 
	# 	else: 
	# 		Malignant_count+=1
	# print("response: ", Malignant_count)
	# print("no response: ", Benign_count)

	# # ## Train set statistics: 
	# # # Malignant:  61
	# # # Benign:  252
	# # ##  Test set statistics:
	# # # Malignant:  16
	# # # Benign:  71
	# # # #  Total:  77 malignant, 323 benign	
	# # #pass 


	classification_dataset_test = Classificaiton_Dataset(phase = 'test')


	Malignant_count = 0 
	Benign_count = 0 

	for i in range(len(classification_dataset_test)):
		if classification_dataset_test[i][2] == 0: 
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
	import random 
	print("test dataset size: ", len(classification_dataset_test))
	for i in random.sample(range(len(classification_dataset_test)), 10):
		img, radiomics, label = classification_dataset_test[i]
		print("radiomics: ", radiomics.shape)
		print(label)
		plt.figure()
		plt.imshow(img[0], cmap= 'gray')
		plt.show()

