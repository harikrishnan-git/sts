import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import cv2

class STSDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def load_middle_slice(self, dicom_dir):
        files = sorted([
            os.path.join(dicom_dir, f)
            for f in os.listdir(dicom_dir)
            if f.endswith(".dcm")
        ])
        dcm = pydicom.dcmread(files[len(files)//2])
        img = dcm.pixel_array.astype(np.float32)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = self.load_middle_slice(row.dicom_dir)
        img = (img - img.mean()) / (img.std() + 1e-5)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img]*3, axis=0)

        label = row["outcome"]   # or grade / histology
        label = 1 if label == "Mets" else 0

        return torch.tensor(img), torch.tensor(label)
