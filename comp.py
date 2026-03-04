import torch
from torch.utils.data import DataLoader
from dataset.mri_dataset import MRIDataset
from dataset.episodic_dataset import EpisodicDataset
from models.ViTContainer import ViTContainer
from train.train_fewshot import train_fewshot
import config
import matplotlib.pyplot as plt
from train.loss_registry import LOSS_REGISTRY

results = {}

base_ds = MRIDataset(config.CSV_PATH)

episodic_ds = EpisodicDataset(
        base_ds,
        config.N_WAY,
        config.K_SHOT,
        config.Q_QUERY,
        config.EPISODES_PER_EPOCH
    )

print("Valid episodic classes:", episodic_ds.valid_sequences)
print("Number of groups: ",len(episodic_ds.valid_sequences))

loader = DataLoader(episodic_ds, batch_size=1, shuffle=True)
print("Data loading complete!!!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTContainer(config.EMBED_DIM)

for loss_name, loss_fn in LOSS_REGISTRY.items():

    print(f"\n====== Training with {loss_name} ======\n")    

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    history = train_fewshot(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        epochs=config.EPOCHS,
        loss_fn=loss_fn,
        loss_name=loss_name
    )

    results[loss_name] = history[-1]

plt.bar(results.keys(), results.values())
plt.title("Few-shot Loss Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Loss Type")
plt.show()
