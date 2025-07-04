import pandas as pd

#1️⃣ Read the main dataset
df = pd.read_csv(r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_sdf4.csv")  # Replace with your actual data
#Ensure it has a 'patientID' column
print(df.head())

# 2️⃣ Read labels from both sheets
labels_sheet1 = pd.read_excel("data/PAT_imaging_record.xlsx", sheet_name="ROI STATS V3", usecols=["Patient ID", "GT"])
labels_sheet2 = pd.read_excel("data/PAT_imaging_record.xlsx", sheet_name="ROI STATS V4 (3)", usecols=["Patient ID", "GT"])

# 3️⃣ Concatenate labels
all_labels = pd.concat([labels_sheet1, labels_sheet2], ignore_index=True)

# 4️⃣ Drop duplicate patientIDs if needed (keeping the LAST one as priority)
all_labels = all_labels.drop_duplicates(subset="Patient ID", keep="last")

# 5️⃣ Perform the merge
# This will match on patientID and repeat the 'label' for every row
# # with the same patientID
# df = df.merge(all_labels, on="PatientID", how="left")

# # ✅ Done! Now every row with the same patientID has the associated label

df = df.merge(all_labels, left_on="PatientID", right_on="Patient ID", how="left")
print(df.head())

df.to_csv(r"Only_radiomics_based_classification\radiomics_features_washu2_p1_143_with_labels_sdf4.csv", index=False)