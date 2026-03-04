import torch
import torch.nn.functional as F


def prototypical_loss_acc(support_emb, support_y, query_emb, query_y):
    classes = torch.unique(support_y)

    label_map = {c.item(): i for i, c in enumerate(classes)}

    support_y_epi = torch.tensor(
        [label_map[y.item()] for y in support_y],
        device=support_y.device
    )

    query_y_epi = torch.tensor(
        [label_map[y.item()] for y in query_y],
        device=query_y.device
    )

    prototypes = []
    for c in range(len(classes)):
        prototypes.append(support_emb[support_y_epi == c].mean(0))
    prototypes = torch.stack(prototypes)

    logits = -torch.cdist(query_emb, prototypes)

    loss = F.cross_entropy(logits, query_y_epi)
    pred = logits.argmax(dim=1)

    acc = (pred == query_y_epi).float().mean()
    return loss, acc
