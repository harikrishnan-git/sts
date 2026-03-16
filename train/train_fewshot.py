import os
import torch
from losses.prototypical_loss import prototypical_loss_acc

_TRAIN_DIR  = os.path.dirname(os.path.abspath(__file__))   # sts/train/
_REPO_ROOT  = os.path.abspath(os.path.join(_TRAIN_DIR, ".."))  # sts/
_WEIGHT_DIR = os.path.join(_REPO_ROOT, "weights")


def train_fewshot(
    model,
    loader,
    optimizer,
    device,
    epochs,
    loss_fn=prototypical_loss_acc,
    loss_name="proto",
    save_path=None                       # if None, defaults to sts/weights/
):
    if save_path is None:
        save_path = _WEIGHT_DIR

    os.makedirs(save_path, exist_ok=True)

    model.to(device)
    history = []

    for ep in range(epochs):
        total_loss = 0.0
        total_acc  = 0.0
        model.train()

        for support_x, support_y, query_x, query_y in loader:
            support_x = support_x.squeeze(0).to(device)
            query_x   = query_x.squeeze(0).to(device)
            support_y = support_y.squeeze(0).to(device)
            query_y   = query_y.squeeze(0).to(device)

            support_emb = model.encode(support_x)
            query_emb   = model.encode(query_x)

            loss, acc = loss_fn(
                support_emb, support_y,
                query_emb,   query_y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += acc.item()

        avg_loss = total_loss / len(loader)
        avg_acc  = total_acc  / len(loader)
        history.append(avg_acc)

        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        ckpt = os.path.join(save_path, f"siamese_vit_fewshot{ep}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Model saved to: {ckpt}")

    return history