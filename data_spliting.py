import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# Step 1: Read Excel sheet
root_dir = r"data\Ovarian_Reviewed_Data"
k_fold = 5
fold = 0 
response_dir = r"data\PAT_imaging_record.xlsx"
df = pd.read_excel(response_dir, sheet_name="ROI STATS V4 (3)")
df = df.dropna(subset=["Patient ID", "Side", "GT"])
df["Patient ID"] = df["Patient ID"].astype(int).astype(str).str.strip()
df["Side"] = df["Side"].astype(str).str.strip()
df["GT"] = pd.to_numeric(df["GT"], errors="coerce").astype("Int64")
df = df.dropna(subset=["GT"])
df["GT"] = (df["GT"].astype(int)<1).astype(int)

# Step 2: Create grouped (Patient ID + Side) level GT map
# This defines each case uniquely by (Patient ID, Side)
df["PatientSide"] = df.apply(lambda row: f"p{row['Patient ID']}_{row['Side']}", axis=1)
grouped_gt = df.groupby("PatientSide")["GT"].agg(lambda x: x.mode()[0])  # majority vote if needed

# Final list of unique (Patient, Side) cases and their GT
case_ids = grouped_gt.index.tolist()
case_labels = grouped_gt.values.tolist()

# Step 3: Stratified split by patient-side cases
skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
splits = list(skf.split(case_ids, case_labels))

train_indices, val_indices = splits[fold]
train_cases = [case_ids[i] for i in train_indices]
val_cases = [case_ids[i] for i in val_indices]

train_cases_set = set(train_cases)
val_cases_set = set(val_cases)

# Step 4: Scan patient directories and match p{PatientID}/{Side}
train_patient_dirs = set()
val_patient_dirs = set()

for patient_id in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for side in os.listdir(patient_path):
        side_path = os.path.join(patient_path, side)
        if not os.path.isdir(side_path):
            continue

        case_key = f"{patient_id}_{side}"
        if case_key in train_cases_set:
            train_patient_dirs.add(side_path)
        elif case_key in val_cases_set:
            val_patient_dirs.add(side_path)

# Final check
assert train_patient_dirs.isdisjoint(val_patient_dirs), "Leakage between train and val sets!"

# Print summary
print(f"Fold {fold} Summary:")
print(f"Train patient-sides: {len(train_patient_dirs)}")
print(f"Val patient-sides:   {len(val_patient_dirs)}")