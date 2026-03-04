import torch
import torch.nn.functional as F
def matchingnet_loss_acc(support_emb, support_y, query_emb, query_y):
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

    support_emb = F.normalize(support_emb, dim=1)
    query_emb   = F.normalize(query_emb, dim=1)

    sim = query_emb @ support_emb.T
    attn = F.softmax(sim, dim=1)

    support_onehot = F.one_hot(support_y_epi, num_classes=len(classes)).float()
    probs = attn @ support_onehot

    loss = F.nll_loss(torch.log(probs + 1e-8), query_y_epi)
    pred = probs.argmax(dim=1)

    acc = (pred == query_y_epi).float().mean()
    return loss, acc
