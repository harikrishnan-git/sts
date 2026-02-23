import os
import re
import pandas as pd
import pydicom
import numpy as np

# ------------------ PATHS ------------------
ROOT = r"/content/Soft-tissue-Sarcoma"
STUDY_LIST_CSV = r"/content/drive/MyDrive/Main project/data/study_list.csv"
OUTPUT_CSV = "/content/drive/MyDrive/Main project/data/sts_index_all_sequences.csv"

# ------------------ LOAD STUDY LIST ------------------
study_df = pd.read_csv(STUDY_LIST_CSV)

PATIENT_COL = "Patient ID"
LABEL_COL = "Histological type"

if PATIENT_COL not in study_df.columns or LABEL_COL not in study_df.columns:
    raise ValueError(f"Missing required columns in study_list.csv")

patient_to_label = (
    study_df[[PATIENT_COL, LABEL_COL]]
    .drop_duplicates(PATIENT_COL)
    .set_index(PATIENT_COL)[LABEL_COL]
    .to_dict()
)

LATERAL_BODY_PARTS = {
    "KNEE", "LEG", "THIGH", "ARM",
    "SHOULDER", "HAND", "FOOT",
    "HIP", "ELBOW", "WRIST","FEMUR",
    "FEMURCUISSE","JAMBE","FMURCUISSE"
}

# ------------------ HELPERS ------------------

def normalize_orientation(name):
    name = name.upper()
    if any(x in name for x in ["AXI", "AXIAL", " AX ","-AX "]):
        return "AXIAL"
    if any(x in name for x in ["COR", "CORONAL"]):
        return "CORONAL"
    if any(x in name for x in ["SAG", "SAGITTAL"]):
        return "SAGITTAL"
    return "UNKNOWN"

def extract_slice_thickness(name):
    match = re.search(r"(\d+\.\d+)", name)
    return float(match.group(1)) if match else "UNKNOWN"

def normalize_sequence(name):
    name = name.lower()
    if "stir" in name:
        return "STIR"
    if "t1" in name and any(x in name for x in ["c", "post", "contrast"]):
        return "T1CE"
    if "t1" in name:
        return "T1"
    if "t2" in name:
        return "T2"
    if "dwi" in name or "diff" in name:
        return "DWI"
    if "adc" in name:
        return "ADC"
    if "flair" in name:
        return "FLAIR"
    if "pd" in name:
        return "PD"
    return "OTHER"

def extract_pulse(name):
    name = name.upper()
    if "FSE" in name:
        return "FSE"
    if "GRE" in name:
        return "GRE"
    if "SE" in name:
        return "SE"
    if "SPGR" in name:
        return "SPGR"
    return "UNKNOWN"

def normalize_laterality(body_part, laterality):
    # lateral anatomy
    if body_part in {
        "KNEE","LEG","THIGH","ARM","SHOULDER",
        "HAND","FOOT","HIP","ELBOW","WRIST","EXTREMITY"
    }:
        return laterality if laterality in {"LEFT","RIGHT"} else "UNKNOWN"

    # midline anatomy
    if body_part in {"PELVIS","SPINE","CHEST","ABDOMEN","BRAIN"}:
        return "NA"

    # generic regions
    if body_part in "WHOLE_BODY":
        return "NA"

    return "UNKNOWN"

def extract_laterality(study_name):
    name = study_name.upper()

    # normalize separators
    name = re.sub(r"[_\-]", " ", name)

    # English
    if re.search(r"\b(RT|RIGHT)\b", name):
        return "RIGHT"
    if re.search(r"\b(LT|LEFT)\b", name):
        return "LEFT"

    # French
    if re.search(r"\bGAUCHE\b", name):
        return "LEFT"
    if re.search(r"\bDROITE\b", name):
        return "RIGHT"

    return "UNKNOWN"

def extract_body_part(study_name):
    name = study_name.upper()

    # Whole-body / PET-CT
    if "PET CT" in name or "TEP TDM" in name:
        return "WHOLE_BODY"

    # French generic limb
    if "EXTREMITE" in name or "EXTR"in name:
        return "EXTREMITY"

    if "FMURCUISSE" in name or "FEMUR" in name or "CUISSE" in name:
      return "THIGH"

    if "BRAS" in name:
      return "ARM"
    
    if "JAMBE" in name:
      return "LEG"

    if "HANCHE" in name or "BUTTOCK" in name or "MSKHIP" in name:
      return "HIP"

    # English specific parts
    for part in [
        "KNEE", "BRAIN", "THIGH", "LEG", "ARM",
        "PELVIS", "ABDOMEN", "CHEST",
        "THORAX","HIP","SPINE"
    ]:
        if part in name:
            return part

    return "UNKNOWN"

def extract_dicom_metadata(dicom_dir):
    meta = {
        "orientation": "UNKNOWN",
        "slice_thickness": "UNKNOWN",
        "sequence_type": "UNKNOWN",
        "pulse": "UNKNOWN",
        "image": True,
        "modality": "UNKNOWN",
    }

    if not os.path.isdir(dicom_dir):
        return meta

    for f in os.listdir(dicom_dir):
        path = os.path.join(dicom_dir, f)
        if not os.path.isfile(path):
            continue

        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue

        # modality
        meta["modality"] = getattr(dcm, "Modality", "UNKNOWN")

        # exclude RTSTRUCT
        sop = str(getattr(dcm, "SOPClassUID", ""))
        if "RT Structure Set" in sop:
            meta["image"] = False

        # slice thickness
        if hasattr(dcm, "SliceThickness") and dcm.SliceThickness is not None:
          try:
              meta["slice_thickness"] = float(dcm.SliceThickness)
          except (TypeError, ValueError):
              pass  # keep UNKNOWN

        # orientation
        if hasattr(dcm, "ImageOrientationPatient"):
            iop = np.array(dcm.ImageOrientationPatient, dtype=float)
            row, col = iop[:3], iop[3:]
            normal = np.abs(np.cross(row, col))

            if normal[2] >= normal[0] and normal[2] >= normal[1]:
                meta["orientation"] = "AXIAL"
            elif normal[1] >= normal[0]:
                meta["orientation"] = "CORONAL"
            else:
                meta["orientation"] = "SAGITTAL"

        # sequence + pulse (best-effort)
        desc = (
            str(getattr(dcm, "SeriesDescription", "")).lower() +
            str(getattr(dcm, "ScanningSequence", "")).lower()
        )

        if "stir" in desc:
            meta["sequence_type"] = "STIR"
        elif "t1" in desc and ("post" in desc or "contrast" in desc):
            meta["sequence_type"] = "T1CE"
        elif "t1" in desc:
            meta["sequence_type"] = "T1"
        elif "t2" in desc:
            meta["sequence_type"] = "T2"

        if hasattr(dcm, "ScanningSequence"):
            if "SE" in dcm.ScanningSequence:
                meta["pulse"] = "SE"
            elif "GR" in dcm.ScanningSequence:
                meta["pulse"] = "GRE"

        break  # one DICOM is enough

    return meta

# ------------------ BUILD INDEX ------------------
rows = []

for patient_id in os.listdir(ROOT):
    patient_path = os.path.join(ROOT, patient_id)
    if not os.path.isdir(patient_path):
        continue

    histo = patient_to_label.get(patient_id, "UNKNOWN")

    for study in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study)
        if not os.path.isdir(study_path):
            continue

        study_upper = study.upper()

        body_part = extract_body_part(study)
        laterality = extract_laterality(study)

        is_pet_ct = any(x in study_upper for x in ["CT PET","PETCT","PET CT", "TEP TDM"])
        is_mri = any(x in study_upper for x in ["IRM", "MRI"])

        # default: assume MR unless PET/CT
        modality = "MR"
        if is_pet_ct:
          modality = "MR_REGISTERED"
        else:
          modality = "MR"   # safe fallback
        if body_part == "WHOLE_BODY":
          final_modality = "MR_REGISTERED"   

        for seq_folder in os.listdir(study_path):
            seq_path = os.path.join(study_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            is_rtstruct = "RTSTRUCT" in seq_folder.upper()

            dicom_meta = extract_dicom_metadata(seq_path)

            orientation = normalize_orientation(seq_folder)
            if orientation == "UNKNOWN":
                orientation = dicom_meta["orientation"]

            slice_thickness = extract_slice_thickness(seq_folder)
            if slice_thickness == "UNKNOWN":
                slice_thickness = dicom_meta["slice_thickness"]

            sequence_type = normalize_sequence(seq_folder)
            if sequence_type == "OTHER":
                sequence_type = dicom_meta["sequence_type"]

            pulse = extract_pulse(seq_folder)
            if pulse == "UNKNOWN":
                pulse = dicom_meta["pulse"]

            rows.append({
                "patient_id": patient_id,
                "histological_type": histo,
                "study_folder": study,
                "body_part": body_part,
                "laterality": normalize_laterality(body_part, laterality),
                "modality": dicom_meta["modality"] if dicom_meta["modality"] != "UNKNOWN" else modality,
                "image": dicom_meta["image"] and not is_rtstruct,
                "sequence_folder": seq_folder,
                "sequence_type": sequence_type,
                "orientation": orientation,
                "slice_thickness": slice_thickness,
                "pulse": pulse,
                "dicom_dir": seq_path
            })


# ------------------ SAVE ------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Index created successfully")

print("\nSequence distribution:")
print(df["sequence_type"].value_counts())

print("\nBody part distribution:")
print(df["body_part"].value_counts())

print("\nOrientation distribution:")
print(df["orientation"].value_counts())

print("\nModality distribution:")
print(df["modality"].value_counts())
