import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from models.ViTContainer import ViTContainer
from dataset.mri_dataset import MRIDataset


# ---------------- CONFIG ----------------
MODEL_PATH = "weights/siamese_vit_fewshot9.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_WAY = 3
K_SHOT = 1
Q_QUERY = 1
EPISODES = 200

OUTPUT_DIR = "eval_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- DATA UTILS ----------------
def build_class_to_indices(dataset):

    class_to_indices = {}

    for i in range(len(dataset)):
        label = dataset.get_label(i)

        if label not in class_to_indices:
            class_to_indices[label] = []

        class_to_indices[label].append(i)

    return class_to_indices


# ---------------- EPISODE SAMPLER ----------------
def sample_episode(dataset, class_to_indices):

    valid_classes = [
        c for c in class_to_indices.keys()
        if len(class_to_indices[c]) >= (K_SHOT + Q_QUERY)
    ]

    selected_classes = np.random.choice(valid_classes, N_WAY, replace=False)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for episode_label, real_class in enumerate(selected_classes):

        indices = np.random.choice(
            class_to_indices[real_class],
            K_SHOT + Q_QUERY,
            replace=False
        )

        support_idx = indices[:K_SHOT]
        query_idx = indices[K_SHOT:]

        for idx in support_idx:
            x, _ = dataset[idx]
            support_x.append(x)
            support_y.append(episode_label)

        for idx in query_idx:
            x, _ = dataset[idx]
            query_x.append(x)
            query_y.append(episode_label)

    return (
        torch.stack(support_x),
        torch.tensor(support_y),
        torch.stack(query_x),
        torch.tensor(query_y)
    )


# ---------------- PROTOTYPE CLASSIFIER ----------------
def proto_classify(model, support_x, support_y, query_x):

    with torch.no_grad():

        support_emb = model.encode(support_x)
        query_emb = model.encode(query_x)

        prototypes = []

        for c in range(N_WAY):
            proto = support_emb[support_y == c].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes)

        logits = torch.matmul(query_emb, prototypes.T)
        preds = torch.argmax(logits, dim=1)

    return preds, query_emb.cpu(), prototypes.cpu()


# ---------------- VISUALIZATION ----------------
def save_confusion_matrix(labels, preds):

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

    return path


# ---------------- MAIN EVAL ----------------
def evaluate():

    dataset = MRIDataset(csv_path="../data/index.csv")

    class_to_indices = build_class_to_indices(dataset)

    model = ViTContainer(embed_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    for _ in tqdm(range(EPISODES)):

        support_x, support_y, query_x, query_y = sample_episode(dataset, class_to_indices)

        support_x = support_x.to(DEVICE)
        query_x = query_x.to(DEVICE)
        query_y = query_y.to(DEVICE)

        preds, _, _ = proto_classify(model, support_x, support_y, query_x)

        correct += (preds.to(DEVICE) == query_y).sum().item()
        total += query_y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(query_y.cpu().numpy())

    acc = correct / total

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    cm_path = save_confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "confusion_matrix_plot": cm_path
    }

    return results


if __name__ == "__main__":

    results = evaluate()

    print("Accuracy:", results["accuracy"])
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])