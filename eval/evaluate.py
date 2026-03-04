import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from models.ViTContainer import ViTContainer
from dataset.mri_dataset import MRIDataset   
from torch.utils.data import DataLoader


# ---------------- CONFIG ----------------
MODEL_PATH = "/workspace/weights/siamese_vit_fewshot9.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# +
N_WAY = 3
K_SHOT = 1
Q_QUERY = 1
EPISODES = 200

OUTPUT_DIR = "/workspace/eval_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    Prototype-based classification using dot product similarity.
    (If you want cosine, normalize embeddings first.)
    """

    with torch.no_grad():
        support_emb = model.encode(support_x)
        query_emb = model.encode(query_x)

        prototypes = []
        for c in range(n_way):
            proto = support_emb[support_y == c].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes)

        logits = torch.matmul(query_emb, prototypes.T)
        preds = torch.argmax(logits, dim=1)

    return preds, query_emb.cpu(), prototypes.cpu()


# ---------------- VISUALIZATION FUNCTIONS ----------------
def save_confusion_matrix(labels, preds, path):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_episode_accuracy_curve(acc_list, path):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_list)
    plt.title("Episode-wise Accuracy Curve")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_per_class_accuracy(labels, preds, path):
    labels = np.array(labels)
    preds = np.array(preds)

    unique_classes = np.unique(labels)
    accs = []

    for c in unique_classes:
        mask = labels == c
        accs.append((preds[mask] == labels[mask]).mean())

    plt.figure(figsize=(8, 5))
    plt.bar([str(c) for c in unique_classes], accs)
    plt.title("Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_tsne_plot(query_embs, query_labels, proto_embs, path):
    """
    query_embs: [total_queries, D]
    proto_embs: [total_prototypes, D]
    """

    all_points = torch.cat([query_embs, proto_embs], dim=0).numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    points_2d = tsne.fit_transform(all_points)

    q_points = points_2d[:len(query_embs)]
    p_points = points_2d[len(query_embs):]

    plt.figure(figsize=(10, 8))

    plt.scatter(q_points[:, 0], q_points[:, 1], c=query_labels, alpha=0.6, label="Query")
    plt.scatter(p_points[:, 0], p_points[:, 1], marker="X", s=200, c="black", label="Prototypes")

    plt.title("t-SNE: Query Embeddings + Prototypes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------------- MAIN EVAL ----------------
def evaluate():
    print("Loading dataset...")
    dataset = MRIDataset(csv_path="/workspace/data/index.csv")

    class_to_indices = build_class_to_indices(dataset)

    print("Loading model...")
    model = ViTContainer(embed_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    episode_accs = []

    all_query_embs = []
    all_query_labels = []

    all_proto_embs = []

    correct = 0
    total = 0

    print(f"Running evaluation for {EPISODES} episodes...")

    for _ in tqdm(range(EPISODES)):
        support_x, support_y, query_x, query_y = sample_episode(
            dataset, class_to_indices, N_WAY, K_SHOT, Q_QUERY
        )

        support_x = support_x.to(DEVICE)
        query_x = query_x.to(DEVICE)
        query_y = query_y.to(DEVICE)

        preds, query_emb, prototypes = proto_classify(model, support_x, support_y, query_x, N_WAY)

        correct_ep = (preds.to(DEVICE) == query_y).sum().item()
        total_ep = query_y.size(0)

        episode_acc = correct_ep / total_ep
        episode_accs.append(episode_acc)

        correct += correct_ep
        total += total_ep

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(query_y.detach().cpu().numpy())

        all_query_embs.append(query_emb)
        all_query_labels.extend(query_y.cpu().numpy())

        all_proto_embs.append(prototypes)

    acc = correct / total

    print("\n================ RESULTS ================")
    print(f"Episodes: {EPISODES}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # ---------------- SAVE VISUALIZATIONS ----------------
    print("\nSaving visualizations...")

    save_confusion_matrix(all_labels, all_preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    save_episode_accuracy_curve(episode_accs, os.path.join(OUTPUT_DIR, "episode_accuracy_curve.png"))
    save_per_class_accuracy(all_labels, all_preds, os.path.join(OUTPUT_DIR, "per_class_accuracy.png"))

    all_query_embs = torch.cat(all_query_embs, dim=0)
    all_proto_embs = torch.cat(all_proto_embs, dim=0)

    save_tsne_plot(
        all_query_embs,
        np.array(all_query_labels),
        all_proto_embs,
        os.path.join(OUTPUT_DIR, "tsne_plot.png")
    )

    print(f"Saved plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate()

