import torch
from losses.prototypical_loss import prototypical_loss_acc


def train_fewshot(
    model,
    loader,
    optimizer,
    device,
    epochs,
    loss_fn = prototypical_loss_acc,
    loss_name="proto",
    save_path="/workspace/weights/"
):
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
                query_emb, query_y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += acc.item()

        avg_loss = total_loss / len(loader)
        avg_acc  = total_acc / len(loader)
        history.append(avg_acc)

        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

        torch.save(model.state_dict(), save_path + f"siamese_vit_fewshot{ep}.pth")
        print(f"Model saved to: {save_path}siamese_vit_fewshot{ep}.pth")
    return history


