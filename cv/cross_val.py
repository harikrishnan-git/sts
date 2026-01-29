import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import os

# -------- CONFIG --------
INDEX_CSV = "data/sts_index_all_sequences.csv"
N_EPISODES = 5
K_FOLDS = 5
OUT_DIR = "episodic_cv"
SEED = 42
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INDEX_CSV)
all_patients = df["patient_id"].unique()

rng = np.random.RandomState(SEED)

for episode in range(N_EPISODES):
    print(f"\n=== Episode {episode} ===")

    # ---- shuffle patients ----
    shuffled_patients = rng.permutation(all_patients)

    # map patient → episode id (for reproducibility)
    patient_map = {p: i for i, p in enumerate(shuffled_patients)}
    df["episode_group"] = df["patient_id"].map(patient_map)

    X = df.index
    y = df["sequence_type"]
    groups = df["episode_group"]

    gkf = GroupKFold(n_splits=K_FOLDS)

    episode_dir = os.path.join(OUT_DIR, f"episode_{episode}")
    os.makedirs(episode_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_df.to_csv(f"{episode_dir}/train_fold_{fold}.csv", index=False)
        val_df.to_csv(f"{episode_dir}/val_fold_{fold}.csv", index=False)

        print(f"Episode {episode} | Fold {fold} | "
              f"Train patients: {train_df.patient_id.nunique()} | "
              f"Val patients: {val_df.patient_id.nunique()}")
