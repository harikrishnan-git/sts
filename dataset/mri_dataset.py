import os
import cv2
from torch.serialization import LoadEndianness
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MRIDataset(Dataset):
    def __init__(self, csv_path, label_col="histological_type"):
        df = pd.read_csv(csv_path)
        self.label_col = label_col

        # ------------------ sanity check ------------------
        required_cols = [
          "dicom_dir",          # where the DICOMs live
          label_col,            # histological type / class label
          "image",              # exclude RTSTRUCT / non-image
          "modality",           # MR vs MR_REGISTERED
          "body_part",          # KNEE / CHEST / BRAIN / etc
          "laterality",         # orientation (left/right/NA)
          "sequence_type",      # T1 / T2 / STIR / ...
          "orientation",        # AXIAL / CORONAL / SAGITTAL
          "slice_thickness"    # e.g. 6.0
          # "pulse"               # type of pulse for mri (fse,se,etc)
        ]

        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        # ------------------ normalize boolean columns ------------------
        # for col in ["MRI", "image"]:
        #     df[col] = df[col].astype(str).str.lower().isin(
        #         ["True"]
        #     )

        # ------------------ KEEP ONLY MRI IMAGE ROWS ------------------
        df = df[
          (df["image"] == True) &
          (df["modality"] == "MR") &
          (df["body_part"] != "UNKNOWN") &
          (df["laterality"] != "UNKNOWN") &
          (df["sequence_type"] != "OTHER") &
          (df["orientation"] != "UNKNOWN") &
          (df["slice_thickness"] != "UNKNOWN") 
        ].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(
                "No usuable rows found in CSV"
            )

        self.df = df

        # ------------------ labels ------------------
        self.classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # ------------------ transforms ------------------
        self.transforms = None

        print(f"MRIDataset initialized with {len(self.df)} MRI image samples")

    def __len__(self):
        return len(self.df)

    def load_middle_slice(self, dicom_dir):
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
          if not hasattr(dcm, "PixelData"):
              return None

          img = dcm.pixel_array.astype(np.float32)
      except Exception:
          return None

      # normalize per-slice
      img = (img - img.mean()) / (img.std() + 1e-8)

      img = cv2.resize(img, (224, 224))

      # SINGLE CHANNEL
      img = img[None, :, :]   # (1, 224, 224)

      return img


    def get_label(self, idx):
      row = self.df.iloc[idx]
      return self.class_to_idx[row[self.label_col]]

    def __getitem__(self, idx):
        # retry logic (very rarely needed now)
        for _ in range(5):
            row = self.df.iloc[idx]
            img = self.load_middle_slice(row["dicom_dir"])

            if img is not None:
                img = torch.from_numpy(img).float()
                label = self.class_to_idx[row[self.label_col]]
                return img, torch.tensor(label, dtype=torch.long)

            idx = np.random.randint(0, len(self.df))

        raise RuntimeError("Failed to load MRI image after retries")
