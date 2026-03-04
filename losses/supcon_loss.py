import torch
import torch.nn.functional as F

def supcon_loss_acc(support_emb, support_y, query_emb, query_y, temperature=0.07):
    """
    support_emb: [N_support, D]
    support_y:   [N_support]
    query_emb:   [N_query, D]
    query_y:     [N_query]

    Returns:
        loss (SupCon), acc (Proto classification accuracy)
    """

    # ---------------- SUPCON LOSS ----------------
    features = torch.cat([support_emb, query_emb], dim=0)
    labels   = torch.cat([support_y, query_y], dim=0)

    features = F.normalize(features, dim=1)
    labels = labels.contiguous()

    sim = features @ features.T
    sim = sim / temperature

    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    labels = labels.unsqueeze(1)
    matches = (labels == labels.T).float()
    matches = matches.masked_fill(mask, 0)

    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (matches * log_prob).sum(dim=1) / (matches.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()

    # ---------------- ACCURACY (PROTO CLASSIFIER) ----------------
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
