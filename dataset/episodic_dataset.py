import random
import torch
from torch.utils.data import Dataset


LATERAL_BODY_PARTS = {
    "KNEE", "LEG", "THIGH", "ARM",
    "SHOULDER", "HAND", "FOOT",
    "HIP", "ELBOW", "WRIST"
}


class EpisodicDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        n_way,
        k_shot,
        q_query,
        episodes,
        label_col="histological_type"
    ):
        self.base = base_dataset
        self.df = base_dataset.df.reset_index(drop=True)

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self.label_col = label_col

        # -------------------------------------------------
        # Build: sequence -> class -> list of index lists
        # -------------------------------------------------
        self.sequence_class_pool = {}

        for i in range(len(self.df)):
            row = self.df.iloc[i]

            # enforce MR only
            if row["modality"] != "MR":
                continue

            body_part = row["body_part"]
            laterality = row["laterality"]
            if laterality == "UNKNOWN":
                continue

            # laterality only when meaningful
            if body_part in LATERAL_BODY_PARTS:
                lat = laterality
            else:
                lat = "NA"

            sequence = row["sequence_type"]

            group_key = (
                body_part,
                lat,
                sequence,
                row["orientation"]
            )

            cls = row[self.label_col]

            self.sequence_class_pool \
                .setdefault(sequence, {}) \
                .setdefault(cls, {}) \
                .setdefault(group_key, []) \
                .append(i)

        # -------------------------------------------------
        # Filter valid (sequence, class, group) combinations
        # -------------------------------------------------
        self.valid_sequences = {}

        for seq, class_dict in self.sequence_class_pool.items():
            valid_class_groups = {}

            for cls, group_dict in class_dict.items():
                valid_groups = [
                    (gk, idxs)
                    for gk, idxs in group_dict.items()
                    if len(idxs) >= (k_shot + q_query)
                ]

                if valid_groups:
                    valid_class_groups[cls] = valid_groups

            if len(valid_class_groups) >= n_way:
                self.valid_sequences[seq] = valid_class_groups

        if not self.valid_sequences:
            raise RuntimeError("❌ No valid episodic sequences found.")

        print(
            f"✅ EpisodicDataset initialized with "
            f"{len(self.valid_sequences)} sequence types"
        )

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        # -------------------------------------------------
        # 1. Pick ONE sequence type for the episode
        # -------------------------------------------------
        sequence = random.choice(list(self.valid_sequences.keys()))
        class_pool = self.valid_sequences[sequence]

        # -------------------------------------------------
        # 2. Pick N classes available in this sequence
        # -------------------------------------------------
        selected_classes = random.sample(
            list(class_pool.keys()), self.n_way
        )

        support_x, support_y = [], []
        query_x, query_y = [], []

        # -------------------------------------------------
        # 3. Sample per class (group-consistent)
        # -------------------------------------------------
        for i, cls in enumerate(selected_classes):
            group_key, idxs = random.choice(class_pool[cls])
            chosen = random.sample(idxs, self.k_shot + self.q_query)

            for j in chosen[:self.k_shot]:
                x, _ = self.base[j]
                support_x.append(x)
                support_y.append(i)

            for j in chosen[self.k_shot:]:
                x, _ = self.base[j]
                query_x.append(x)
                query_y.append(i)

        return (
            torch.stack(support_x),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x),
            torch.tensor(query_y, dtype=torch.long),
        )
