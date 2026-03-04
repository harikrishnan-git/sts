import torch
import torch.nn.functional as F
import random

def triplet_loss_acc(support_emb, support_y, query_emb, query_y, margin=1.0, num_triplets=64):
    """
    support_emb: [N_support, D]
    support_y:   [N_support]
    query_emb:   [N_query, D]
    query_y:     [N_query]

    Returns:
        loss, acc
    """

    # ---------------- TRIPLET LOSS (from support set) ----------------
    y = support_y.cpu().tolist()
    triplets = []

    for _ in range(num_triplets):
        anchor_idx = random.randrange(len(y))
        anchor_label = y[anchor_idx]

        pos_indices = [i for i, lab in enumerate(y) if lab == anchor_label and i != anchor_idx]
        neg_indices = [i for i, lab in enumerate(y) if lab != anchor_label]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        pos_idx = random.choice(pos_indices)
        neg_idx = random.choice(neg_indices)

        triplets.append((anchor_idx, pos_idx, neg_idx))

    if len(triplets) == 0:
        loss = torch.tensor(0.0, device=support_emb.device)
    else:
        a = torch.stack([support_emb[i] for i, _, _ in triplets])
        p = torch.stack([support_emb[j] for _, j, _ in triplets])
        n = torch.stack([support_emb[k] for _, _, k in triplets])

        d_pos = F.pairwise_distance(a, p)
        d_neg = F.pairwise_distance(a, n)

        loss = torch.relu(d_pos - d_neg + margin).mean()

    # ---------------- ACCURACY (prototype classification) ----------------
    classes = torch.unique(support_y)
    support_y_epi = torch.searchsorted(classes, support_y)
    query_y_epi   = torch.searchsorted(classes, query_y)

    prototypes = torch.stack([
        support_emb[support_y_epi == c].mean(dim=0)
        for c in range(len(classes))
    ])

    logits = -torch.cdist(query_emb, prototypes)
    preds = logits.argmax(dim=1)

    acc = (preds == query_y_epi).float().mean()

    return loss, acc
