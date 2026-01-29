import os
import pandas as pd

ROOT = r"/mnt/c/Users/harib/OneDrive/Desktop/Projects/Main project/manifest-MjbMt99Q1553106146386120388/Soft-tissue-Sarcoma"

rows = []

for patient_id in os.listdir(ROOT):
    patient_path = os.path.join(ROOT, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for study in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study)
        if not os.path.isdir(study_path):
            continue

        # Extract body part safely
        parts = study.split("-")
        body_part = "UNKNOWN"
        for p in parts:
            if p.isalpha() and len(p) > 3:
                body_part = p
                break

        for seq_folder in os.listdir(study_path):
            seq_path = os.path.join(study_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            # ---- MRI SEQUENCE HANDLING ----
            seq_name = seq_folder.lower()

            if "t1" in seq_name:
                sequence_type = "T1"
            elif "t2" in seq_name:
                sequence_type = "T2"
            elif "stir" in seq_name:
                sequence_type = "STIR"
            elif "dwi" in seq_name or "diff" in seq_name:
                sequence_type = "DWI"
            elif "adc" in seq_name:
                sequence_type = "ADC"
            elif "flair" in seq_name:
                sequence_type = "FLAIR"
            elif "pd" in seq_name:
                sequence_type = "PD"
            elif "gre" in seq_name:
                sequence_type = "GRE"
            elif "swi" in seq_name:
                sequence_type = "SWI"
            elif "t1c" in seq_name or "post" in seq_name or "contrast" in seq_name:
                sequence_type = "T1CE"
            else:
                sequence_type = "OTHER"

            rows.append({
                "patient_id": patient_id,
                "study_folder": study,
                "body_part": body_part,
                "sequence_folder": seq_folder,   # exact name (IMPORTANT)
                "sequence_type": sequence_type, # normalized label
                "dicom_dir": seq_path
            })

df = pd.DataFrame(rows)
df.to_csv("sts_index_all_sequences.csv", index=False)

print("Index created successfully")
print(df["sequence_type"].value_counts())
