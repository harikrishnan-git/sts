import torch
from losses.contrastive import prototypical_loss


def train_fewshot(
    model,
    loader,
    optimizer,
    device,
    epochs,
    save_path="/workspace/weights/"  
):
    model.to(device)

    for ep in range(epochs):
        total_loss = 0
        model.train()

        for support_x, support_y, query_x, query_y in loader:
            support_x = support_x.squeeze(0).to(device)
            query_x = query_x.squeeze(0).to(device)
            support_y = support_y.squeeze(0).to(device)
            query_y = query_y.squeeze(0).to(device)

            support_emb = model.encode(support_x)
            query_emb = model.encode(query_x)

            loss = prototypical_loss(
                support_emb, support_y,
                query_emb, query_y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), save_path+"siamese_vit_fewshot"+str(ep)+".pth")
        print(f"Model saved to: {save_path}"+"siamese_vit_fewshot"+str(ep)+".pth")

    # SAVE MODEL AFTER TRAINING

