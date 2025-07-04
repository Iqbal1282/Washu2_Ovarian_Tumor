import pandas as pd
import os

# Load both CSVs
df1 = pd.read_csv("Only_radiomics_based_classification/radiomics_features_washu2_p1_143_with_labels.csv")
df2 = pd.read_csv("Only_radiomics_based_classification/radiomics_features_washu2_p1_143_with_labels_sdf4.csv")

# Step 1: Extract matching suffix (e.g., p10/R/1.jpg) for both DataFrames
df1['RelativePath'] = df1['ImagePath'].apply(lambda x: os.path.normpath(x).split('Images')[-1].lstrip(os.sep))
df2['RelativePath'] = df2['ImagePath'].apply(lambda x: os.path.normpath(x).split('Images')[-1].lstrip(os.sep))

# Step 2: Remove duplicates from df2 except the new features
#columns_to_exclude = ['PatientID', 'GT', 'ImagePath']
# Identify non-duplicate columns in df2 to avoid duplicating key fields
columns_to_exclude = ['PatientID', 'GT', 'ImagePath', 'Patient ID']

df2_unique = df2.drop(columns=[col for col in columns_to_exclude if col in df2.columns])

# Step 3: Merge on the cleaned relative path
df_merged = pd.merge(df1, df2_unique, on='RelativePath', how='inner')

# Optional: Drop the extra 'RelativePath' if you don't want to keep it
df_merged.drop(columns=['RelativePath'], inplace=True)

# ✅ Done
print(f"✅ Merged shape: {df_merged.shape}")
# df_merged.to_csv("merged_features_relative_match.csv", index=False)


print(df_merged.head())

df_merged.to_csv("Only_radiomics_based_classification/radiomics_features_washu2_p1_143_with_labels_sdf4_nd_normseg.csv")

