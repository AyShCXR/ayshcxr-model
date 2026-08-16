# make_nih_clean.py — builds nih_clean.csv for train_model.py from nih_full_labels.csv
# (renames columns to what train_model.py expects; reuses the resized image paths).
import pandas as pd
df = pd.read_csv("/workspace/nih_full_labels.csv")
df = df.rename(columns={"Patient ID": "patient_id", "Pleural Thickening": "Pleural_Thickening"})
labels = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
          "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
          "Fibrosis","Pleural_Thickening","Hernia"]
df[["full_path","patient_id"] + labels].to_csv("/workspace/nih_clean.csv", index=False)
print(f"wrote /workspace/nih_clean.csv | {len(df):,} rows")
print("example path:", df["full_path"].iloc[0])
