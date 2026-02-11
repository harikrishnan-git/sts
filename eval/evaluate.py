import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from models.siamese import SiameseViT
from dataset.mri_dataset import MRIDataset   
from torch.utils.data import DataLoader


# ---------------- CONFIG ----------------
MODEL_PATH = "/workspace/weights/siamese_vit_fewshot9.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_WAY = 3
K_SHOT = 1
Q_QUERY = 1
EPISODES = 200   # increase for better evaluation stability

def build_class_to_indices(dataset):
    class_to_indices = {}

    for i in range(len(dataset)):
        label = dataset.get_label(i)
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)

    return class_to_indices

# ---------------- EPISODE SAMPLER ----------------
def sample_episode(dataset, class_to_indices, n_way, k_shot, q_query):

    valid_classes = [c for c in class_to_indices.keys()
                     if len(class_to_indices[c]) >= (k_shot + q_query)]

    selected_classes = np.random.choice(valid_classes, n_way, replace=False)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for episode_label, real_class in enumerate(selected_classes):
        indices = np.random.choice(
            class_to_indices[real_class],
            k_shot + q_query,
            replace=False
        )

        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:]

        for idx in support_idx:
            x, _ = dataset[idx]
            support_x.append(x)
            support_y.append(episode_label)

        for idx in query_idx:
            x, _ = dataset[idx]
            query_x.append(x)
            query_y.append(episode_label)

    support_x = torch.stack(support_x)
    query_x = torch.stack(query_x)

    support_y = torch.tensor(support_y)
    query_y = torch.tensor(query_y)

    return support_x, support_y, query_x, query_y


# ---------------- PROTOTYPE CLASSIFIER ----------------
def proto_classify(model, support_x, support_y, query_x, n_way):
    """
    Prototype-based classification using cosine similarity
    """

    with torch.no_grad():
        support_emb = model.encode(support_x)  # (n_way*k_shot, embed_dim)
        query_emb = model.encode(query_x)      # (n_way*q_query, embed_dim)

        prototypes = []
        for c in range(n_way):
            proto = support_emb[support_y == c].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes)  # (n_way, embed_dim)

        # cosine similarity
        logits = torch.matmul(query_emb, prototypes.T)
        preds = torch.argmax(logits, dim=1)

    return preds


# ---------------- MAIN EVAL ----------------
def evaluate():
    print("Loading dataset...")
    dataset = MRIDataset(csv_path = "/workspace/data/index.csv")  
    
    class_to_indices = build_class_to_indices(dataset)

    print("Loading model...")
    model = SiameseViT(embed_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print(f"Running evaluation for {EPISODES} episodes...")

    for _ in tqdm(range(EPISODES)):
        support_x, support_y, query_x, query_y = sample_episode(dataset, class_to_indices, N_WAY, K_SHOT, Q_QUERY)

        support_x = support_x.to(DEVICE)
        query_x = query_x.to(DEVICE)
        query_y = query_y.to(DEVICE)

        preds = proto_classify(model, support_x, support_y, query_x, N_WAY)

        correct += (preds == query_y).sum().item()
        total += query_y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(query_y.cpu().numpy())

    acc = correct / total
    print("\n================ RESULTS ================")
    print(f"Episodes: {EPISODES}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
