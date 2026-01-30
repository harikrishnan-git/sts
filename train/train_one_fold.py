import torch
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16
import sys
sys.path.append("/content/drive/MyDrive/Main project")
from dataset.sts_dataset import STSDataset

def train_one_fold(train_csv, val_csv, epochs=10):
    train_ds = STSDataset(train_csv)
    val_ds   = STSDataset(val_csv)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8)

    model = vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(768, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        print(f"Epoch {ep} done")

    return model
