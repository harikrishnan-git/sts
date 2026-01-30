import os
import pandas as pd

# ------------------ PATHS ------------------
ROOT = r"/content/drive/MyDrive/Main project/data/Soft-tissue-Sarcoma"
STUDY_LIST_CSV = r"/content/drive/MyDrive/Main project/data/study_list.csv"   # <-- update path if needed
OUTPUT_CSV = "/content/drive/MyDrive/Main project/data/sts_index_all_sequences.csv"

# ------------------ LOAD STUDY LIST ------------------
study_df = pd.read_csv(STUDY_LIST_CSV)

PATIENT_COL = "Patient ID"
LABEL_COL = "Histological type"

# sanity check
if PATIENT_COL not in study_df.columns or LABEL_COL not in study_df.columns:
    raise ValueError(
        f"study_list.csv must contain columns "
        f"'{PATIENT_COL}' and '{LABEL_COL}'. "
        f"Found: {list(study_df.columns)}"
    )

# create patient → histopathology map (unique per patient)
patient_to_label = (
    study_df[[PATIENT_COL, LABEL_COL]]
    .drop_duplicates(subset=PATIENT_COL)
    .set_index(PATIENT_COL)[LABEL_COL]
    .to_dict()
)

# ------------------ BUILD INDEX ------------------
rows = []

for patient_id in os.listdir(ROOT):
    patient_path = os.path.join(ROOT, patient_id)
    if not os.path.isdir(patient_path):
        continue

    # get histopathology label (or UNKNOWN)
    histo_type = patient_to_label.get(patient_id, "UNKNOWN")

    for study in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study)
        if not os.path.isdir(study_path):
            continue

        # -------- BODY PART EXTRACTION --------
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

            # -------- MRI SEQUENCE HANDLING --------
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
                "histological_type": histo_type,   # ✅ from study_list.csv
                "study_folder": study,
                "body_part": body_part,
                "sequence_folder": seq_folder,
                "sequence_type": sequence_type,
                "dicom_dir": seq_path
            })

# ------------------ SAVE ------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Index created successfully")
print("\nSequence distribution:")
print(df["sequence_type"].value_counts())

print("\nHistopathology distribution:")
print(df["histological_type"].value_counts())
