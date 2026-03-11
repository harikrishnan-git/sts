import torch
import pandas as pd
import numpy as np
import os
import cv2
import pydicom

from models.ViTContainer import ViTContainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = r"C:\Users\harib\OneDrive\Desktop\sts\data\index.csv"
SAVE_PATH = "embeddings/support_embeddings.pt"


# -------------------------
# Load CSV
# -------------------------

df = pd.read_csv(CSV_PATH)

df = df.rename(columns={
    "histological_type": "label"
})

print("Dataset size:", len(df))


# -------------------------
# Filter valid MRI images
# -------------------------

df = df[
    (df["image"] == True) &
    (df["modality"] == "MR") &
    (df["sequence_type"] != "OTHER")
].reset_index(drop=True)

print("Filtered MRI sequences:", len(df))


# -------------------------
# Load Model
# -------------------------

model = ViTContainer(embed_dim=256)

model.load_state_dict(
    torch.load("weights/siamese_vit_fewshot9.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()


# -------------------------
# Load middle slice
# -------------------------

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

    return img.to(DEVICE)


# -------------------------
# Compute embeddings
# -------------------------

embeddings = []
labels = []

for _, row in df.iterrows():

    dicom_dir = row["dicom_dir"]
    label = row["label"]

    img = load_middle_slice(dicom_dir)

    if img is None:
        print("Skipping:", dicom_dir)
        continue

    with torch.no_grad():
        emb = model.encode(img)

    embeddings.append(emb.cpu())
    labels.append(label)


embeddings = torch.cat(embeddings)


# -------------------------
# Save embeddings
# -------------------------

os.makedirs("embeddings", exist_ok=True)

torch.save(
    {
        "embeddings": embeddings,
        "labels": labels
    },
    SAVE_PATH
)

print("Support embeddings saved!")
print("Total embeddings:", embeddings.shape)