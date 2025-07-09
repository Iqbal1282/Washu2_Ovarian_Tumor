import pandas as pd
import re

# Example: Load your data
radiomics_df = pd.read_csv("radiomics.csv")   # Contains ImagePath and features
gt_df = pd.read_csv("gt.csv")                 # Contains GT (label), Patient ID, Side

# --- Step 1: Extract patient_id and side from ImagePath ---
def extract_pid_side(path):
    # Example path: data/.../p10/R/1.jpg → extract '10' and 'R'
    match = re.search(r"/p(\d+)/([RC])/", path)
    if match:
        pid = int(match.group(1))
        side = match.group(2)
        return pd.Series({'PatientID': pid, 'Side': side})
    else:
        return pd.Series({'PatientID': None, 'Side': None})

radiomics_df[['PatientID', 'Side']] = radiomics_df['ImagePath'].apply(extract_pid_side)

# --- Step 2: Merge with GT dataframe ---
# Ensure GT DataFrame has correct column names
gt_df.columns = ['GT', 'PatientID', 'Side']
gt_df['PatientID'] = gt_df['PatientID'].astype(int)  # Ensure same dtype

# Merge based on PatientID and Side
merged_df = pd.merge(radiomics_df, gt_df, on=['PatientID', 'Side'], how='left')

# Optional: Drop rows with missing GT (i.e., no match found)
merged_df = merged_df.dropna(subset=['GT'])

# --- Result ---
print(merged_df[['ImagePath', 'PatientID', 'Side', 'GT']].head())  # sample output
