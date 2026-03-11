import torch
import pandas as pd
import numpy as np
import os
import cv2
import pydicom
import torch.nn.functional as F

from models.ViTContainer import ViTContainer
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# PATHS
# ------------------------------------------------

INDEX_CSV = r"C:\Users\harib\OneDrive\Desktop\Projects\Main project\data\sts_index_all_sequences.csv"
CLASS_CSV = r"C:\Users\harib\OneDrive\Desktop\sts\data\study_list.csv"

MODEL_PATH = "weights/siamese_vit_fewshot9.pth"
EMBED_PATH = "embeddings/support_embeddings.pt"

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

model = ViTContainer(config.EMBED_DIM)

model.load_state_dict(
    torch.load("weights/siamese_vit_fewshot9.pth", map_location=device)
)

model = model.to(device)
model.eval()

# ------------------------------------------------
# MRI PREPROCESSING (same as MRIDataset)
# ------------------------------------------------

def load_middle_slice(dicom_dir):

    files = sorted([
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if f.lower().endswith(".dcm")
    ])

    if len(files) == 0:
        return None

    dcm_path = files[len(files)//2]

    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
    except:
        return None

    img = (img - img.mean()) / (img.std() + 1e-8)

    img = cv2.resize(img, (224,224))

    img = img[None,:,:]

    img = torch.from_numpy(img).float().unsqueeze(0)

    return img.to(device)

# ------------------------------------------------
# LOAD CSV DATA
# ------------------------------------------------

def load_dataset():

    index_df = pd.read_csv(INDEX_CSV)

    class_df = pd.read_csv(CLASS_CSV)

    class_df = class_df.rename(columns={
        "Patient ID":"patient_id",
        "Histological type":"label"
    })

    df = index_df.merge(class_df, on="patient_id")

    print("Merged dataset size:", len(df))

    return df

# ------------------------------------------------
# BUILD SUPPORT EMBEDDINGS
# ------------------------------------------------

def build_support_embeddings(df):

    support_embeddings = {}

    grouped = df.groupby("label")

    for label, group in grouped:

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

        prototype = embeddings.mean(0)

        support_embeddings[label] = prototype.cpu()

        print(f"{label} -> {len(embeddings)} samples")

    return support_embeddings

# ------------------------------------------------
# LOAD OR BUILD EMBEDDINGS
# ------------------------------------------------

def get_support_embeddings():

    if os.path.exists(EMBED_PATH):

        print("Loading saved embeddings...")

        data = torch.load(EMBED_PATH)

        embeddings = data["embeddings"]
        labels = data["labels"]

        support_embeddings = {}

        for label in set(labels):

            idx = [i for i,l in enumerate(labels) if l == label]

            class_emb = embeddings[idx]

            prototype = class_emb.mean(0)

            support_embeddings[label] = prototype

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

        emb = emb.to(device)

        score = F.cosine_similarity(query_emb, emb.unsqueeze(0)).item()

        if score > best_score:

            best_score = score
            best_class = label

    return best_class, best_score

# ------------------------------------------------
# RUN
# ------------------------------------------------

support_embeddings = get_support_embeddings()

print("Available classes:", list(support_embeddings.keys()))

query_dir = r"C:\Users\harib\OneDrive\Desktop\sts\data\test-img\STS_034\09-28-2002-NA-MSKHIP-72755\4.000000-AXIAL STIR-50167"
pred, score = predict(query_dir, support_embeddings)

print("\nPrediction:", pred)
print("Similarity:", score)