import sys
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))   # sts/
sys.path.append(_REPO_ROOT)

import torch
import pandas as pd
import numpy as np
import cv2
import pydicom

from models.ViTContainer import ViTContainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# PATHS  (all anchored to repo root)
# ------------------------------------------------

CSV_PATH  = os.path.join(_REPO_ROOT, "data", "index.csv")
SAVE_PATH = os.path.join(_REPO_ROOT, "embeddings", "support_embeddings.pt")
MODEL_PATH = os.path.join(_REPO_ROOT, "weights", "siamese_vit_fewshot9.pth")

# Root of the actual DICOM data on this machine
DICOM_ROOT = r"C:\Users\user\Desktop\dataset\Soft-tissue-Sarcoma"

# Harib's original prefix in the CSV dicom_dir column
_HARIB_PREFIX = "/mnt/c/Users/harib/OneDrive/Desktop/Projects/Main project/manifest-MjbMt99Q1553106146386120388/Soft-tissue-Sarcoma"

def remap_dicom_dir(raw_path: str) -> str:
    """Replace harib's hardcoded prefix with the local DICOM_ROOT."""
    raw_path = raw_path.replace("/", os.sep)
    harib    = _HARIB_PREFIX.replace("/", os.sep)
    if raw_path.startswith(harib):
        return DICOM_ROOT + raw_path[len(harib):]
    return raw_path

# ------------------------------------------------
# Load + filter CSV
# ------------------------------------------------

df = pd.read_csv(CSV_PATH)

df["dicom_dir"] = df["dicom_dir"].apply(remap_dicom_dir)

df = df.rename(columns={"histological_type": "label"})

print("Dataset size:", len(df))

df = df[
    (df["image"]         == True) &
    (df["modality"]      == "MR") &
    (df["sequence_type"] != "OTHER")
].reset_index(drop=True)

print("Filtered MRI sequences:", len(df))

# ------------------------------------------------
# Load model
# ------------------------------------------------

model = ViTContainer(embed_dim=256)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------------------------------------
# Load middle slice
# ------------------------------------------------

def load_middle_slice(dicom_dir):

    files = sorted([
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm")
    ])

    if len(files) == 0:
        return None

    dcm_path = files[len(files) // 2]

    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
    except Exception:
        return None

    img = (img - img.mean()) / (img.std() + 1e-8)
    img = cv2.resize(img, (224, 224))
    img = img[None, :, :]
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img.to(DEVICE)

# ------------------------------------------------
# Compute embeddings
# ------------------------------------------------

embeddings = []
labels     = []

for _, row in df.iterrows():

    dicom_dir = row["dicom_dir"]
    label     = row["label"]

    if not os.path.exists(dicom_dir):
        print("Skipping (path not found):", dicom_dir)
        continue

    img = load_middle_slice(dicom_dir)

    if img is None:
        print("Skipping (load failed):", dicom_dir)
        continue

    with torch.no_grad():
        emb = model.encode(img)

    embeddings.append(emb.cpu())
    labels.append(label)

if len(embeddings) == 0:
    print("No embeddings computed — check DICOM_ROOT path and CSV.")
else:
    embeddings = torch.cat(embeddings)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    torch.save({"embeddings": embeddings, "labels": labels}, SAVE_PATH)

    print("Support embeddings saved to:", SAVE_PATH)
    print("Total embeddings:", embeddings.shape)