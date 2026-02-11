import os
import shutil
import pandas as pd
from tqdm import tqdm

# ================== CONFIG ==================
EXCEL_PATH = r"study_list.csv"

MANIFEST_DIR = r"C:\Users\harib\OneDrive\Desktop\Projects\Main project\manifest-MjbMt99Q1553106146386120388\Soft-tissue-Sarcoma"

OUTPUT_DIR = r"C:\Users\harib\OneDrive\Desktop\Projects\Main project\dataset"

PATIENT_COL = "STSid"        # verify in Excel
LABEL_COL = "Histology"      # or Diagnosis / SarcomaSubtype
# ============================================

# Class mapping
CLASS_MAP = {
    "leiomyo": "class0_leiomyosarcoma",
    "lipo": "class1_liposarcoma"
}
DEFAULT_CLASS = "class2_others"

# Create output dirs
os.makedirs(OUTPUT_DIR, exist_ok=True)
for c in list(CLASS_MAP.values()) + [DEFAULT_CLASS]:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

# Load Excel
df = pd.read_excel(EXCEL_PATH)
df = pd.read_excel(EXCEL_PATH)
print("Columns in Excel:")
print(df.columns.tolist())
exit()

df[PATIENT_COL] = df[PATIENT_COL].astype(str).str.strip()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower()

# Build patient → class mapping
patient_to_class = {}

for _, row in df.iterrows():
    pid = row[PATIENT_COL]
    label = row[LABEL_COL]

    assigned = DEFAULT_CLASS
    for key, class_name in CLASS_MAP.items():
        if key in label:
            assigned = class_name
            break

    patient_to_class[pid] = assigned

# Walk through each STS_xxx folder
for patient_id in tqdm(os.listdir(MANIFEST_DIR), desc="Processing patients"):
    patient_path = os.path.join(MANIFEST_DIR, patient_id)

    if not os.path.isdir(patient_path):
        continue

    target_class = patient_to_class.get(patient_id, DEFAULT_CLASS)
    target_dir = os.path.join(OUTPUT_DIR, target_class)

    # 🔑 Recursive DICOM search (this handles your deep path)
    for root, _, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith(".dcm"):
                src = os.path.join(root, file)

                # Avoid overwrite
                new_name = f"{patient_id}_{os.path.basename(root)}_{file}"
                dst = os.path.join(target_dir, new_name)

                shutil.copy2(src, dst)

print("\n✅ Conversion finished successfully!")
