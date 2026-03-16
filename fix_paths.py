"""
Run this ONCE to fix all dicom_dir paths in index.csv.
Replaces harib's hardcoded paths with your local DICOM root.
"""
import pandas as pd
import os

CSV_PATH   = r"C:\Users\user\sts\data\index.csv"
DICOM_ROOT = r"C:\Users\user\Desktop\dataset\Soft-tissue-Sarcoma"

# All known harib prefixes found in the CSV
HARIB_PREFIXES = [
    r"C:\Users\harib\OneDrive\Desktop\sts\data\Soft-tissue-Sarcoma",
    r"C:\Users\harib\OneDrive\Desktop\Projects\Main project\manifest-MjbMt99Q1553106146386120388\Soft-tissue-Sarcoma",
    "/mnt/c/Users/harib/OneDrive/Desktop/sts/data/Soft-tissue-Sarcoma",
    "/mnt/c/Users/harib/OneDrive/Desktop/Projects/Main project/manifest-MjbMt99Q1553106146386120388/Soft-tissue-Sarcoma",
]

df = pd.read_csv(CSV_PATH)

print(f"Total rows: {len(df)}")
print(f"Sample before: {df['dicom_dir'].iloc[0]}")

def remap(path):
    if not isinstance(path, str):
        return path
    # Normalise slashes for comparison
    normalised = path.replace("/", "\\")
    for prefix in HARIB_PREFIXES:
        norm_prefix = prefix.replace("/", "\\")
        if normalised.startswith(norm_prefix):
            remainder = normalised[len(norm_prefix):]
            return DICOM_ROOT + remainder
    return path  # already correct

df["dicom_dir"] = df["dicom_dir"].apply(remap)

print(f"Sample after:  {df['dicom_dir'].iloc[0]}")

# Save back
df.to_csv(CSV_PATH, index=False)
print("index.csv updated successfully.")

# Verify — check how many paths now exist on disk
exists     = df["dicom_dir"].apply(os.path.exists).sum()
not_exists = len(df) - exists
print(f"Paths that exist on disk:     {exists}")
print(f"Paths that don't exist:       {not_exists}")
if not_exists > 0:
    print("Sample missing paths:")
    missing = df[~df["dicom_dir"].apply(os.path.exists)]["dicom_dir"]
    for p in missing.head(3):
        print(" ", p)