import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import cv2
import torchvision.transforms as T


class STSDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        # ViT / ImageNet-compatible transforms
        self.transforms = T.Compose([
            T.ToTensor(),  # converts HWC → CHW and float32
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # infer number of classes automatically
        self.classes = sorted(self.df["outcome"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.df)

    def load_middle_slice(self, dicom_dir):
        files = sorted([
            os.path.join(dicom_dir, f)
            for f in os.listdir(dicom_dir)
            if f.endswith(".dcm")
        ])

        dcm = pydicom.dcmread(files[len(files) // 2])
        img = dcm.pixel_array.astype(np.float32)

        # scale to [0, 255]
        img = img - img.min()
        img = img / (img.max() + 1e-8)
        img = (img * 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load MRI slice
        img = self.load_middle_slice(row["dicom_dir"])

        # resize to ViT input
        img = cv2.resize(img, (224, 224))

        # grayscale → 3 channel
        img = np.stack([img] * 3, axis=-1)  # HWC

        # transform
        img = self.transforms(img)

        # label
        label = self.class_to_idx[row["histological_type"]]
        label = torch.tensor(label, dtype=torch.long)

        return img, label
