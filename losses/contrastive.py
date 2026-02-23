import torch
import torch.nn.functional as F


def prototypical_loss(support_emb, support_y, query_emb, query_y):
    # get unique classes in THIS episode
    classes = torch.unique(support_y)

    # map global labels → episodic labels (0..N_way-1)
    label_map = {c.item(): i for i, c in enumerate(classes)}

    support_y_epi = torch.tensor(
        [label_map[y.item()] for y in support_y],
        device=support_y.device
    )

    query_y_epi = torch.tensor(
        [label_map[y.item()] for y in query_y],
        device=query_y.device
    )

    # compute prototypes
    prototypes = []
    for c in range(len(classes)):
        prototypes.append(
            support_emb[support_y_epi == c].mean(0)
        )

    prototypes = torch.stack(prototypes)          # [N_way, D]

    # distances → logits
    dists = torch.cdist(query_emb, prototypes)    # [N_query, N_way]
    logits = -dists

    return F.cross_entropy(logits, query_y_epi)
