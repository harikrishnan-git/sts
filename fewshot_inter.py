import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # sts/
sys.path.append(_SCRIPT_DIR)

import torch
import pandas as pd
import numpy as np
import cv2
import pydicom
import torch.nn.functional as F

from models.ViTContainer import ViTContainer
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# PATHS  (all anchored to repo root via config)
# ------------------------------------------------

_REPO_ROOT = _SCRIPT_DIR   # fewshot_inter.py lives in sts/

INDEX_CSV  = os.path.join(_REPO_ROOT, "data", "index.csv")
MODEL_PATH = os.path.join(_REPO_ROOT, "weights", "siamese_vit_fewshot9.pth")
EMBED_PATH = os.path.join(_REPO_ROOT, "embeddings", "support_embeddings.pt")

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
    return raw_path   # already correct or unknown format

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = ViTContainer(config.EMBED_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ------------------------------------------------
# MRI PREPROCESSING  (same as MRIDataset)
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

    return img.to(device)

# ------------------------------------------------
# LOAD CSV DATA
# ------------------------------------------------

def load_dataset():
    df = pd.read_csv(INDEX_CSV)
    df["dicom_dir"] = df["dicom_dir"].apply(remap_dicom_dir)
    df = df.rename(columns={"histological_type": "label"})
    print("Dataset size:", len(df))
    return df

# ------------------------------------------------
# BUILD SUPPORT EMBEDDINGS
# ------------------------------------------------

def build_support_embeddings(df):

    support_embeddings = {}

    for label, group in df.groupby("label"):

        embeddings = []

        for _, row in group.iterrows():

            dicom_dir = row["dicom_dir"]

            if not os.path.exists(dicom_dir):
                continue

            img = load_middle_slice(dicom_dir)

            if img is None:
                continue

            with torch.no_grad():
                emb = model(img)

            embeddings.append(emb)

        if len(embeddings) == 0:
            continue

        embeddings = torch.cat(embeddings)
        prototype  = embeddings.mean(0)
        support_embeddings[label] = prototype.cpu()

        print(f"{label} -> {len(embeddings)} samples")

    return support_embeddings

# ------------------------------------------------
# LOAD OR BUILD EMBEDDINGS
# ------------------------------------------------

def get_support_embeddings():

    if os.path.exists(EMBED_PATH):

        print("Loading saved embeddings...")
        data   = torch.load(EMBED_PATH)
        embeddings = data["embeddings"]
        labels     = data["labels"]

        support_embeddings = {}
        for label in set(labels):
            idx       = [i for i, l in enumerate(labels) if l == label]
            class_emb = embeddings[idx]
            support_embeddings[label] = class_emb.mean(0)

    else:

        print("Building embeddings (first run)...")
        df = load_dataset()
        support_embeddings = build_support_embeddings(df)
        torch.save(support_embeddings, EMBED_PATH)
        print("Embeddings saved to:", EMBED_PATH)

    return support_embeddings

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

def predict(query_dir, support_embeddings):

    img = load_middle_slice(query_dir)

    with torch.no_grad():
        query_emb = model.encode(img)

    best_class = None
    best_score = -1

    for label, emb in support_embeddings.items():
        emb   = emb.to(device)
        score = F.cosine_similarity(query_emb, emb.unsqueeze(0)).item()
        if score > best_score:
            best_score = score
            best_class = label

    return best_class, best_score

# ------------------------------------------------
# RUN
# ------------------------------------------------

if __name__ == "__main__":

    support_embeddings = get_support_embeddings()
    print("Available classes:", list(support_embeddings.keys()))

    # Replace this with your actual query folder
    query_dir = input("Enter query DICOM folder path: ").strip()
    pred, score = predict(query_dir, support_embeddings)

    print("\nPrediction:", pred)
    print("Similarity:", score)