import torch
from torch.utils.data import DataLoader
from dataset.mri_dataset import MRIDataset
from dataset.episodic_dataset import EpisodicDataset
from models.siamese import SiameseViT
from train.train_fewshot import train_fewshot
import config


def main():
    base_ds = MRIDataset("/content/drive/MyDrive/Main project/data/sts_index_all_sequences.csv")

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

    model = SiameseViT(config.EMBED_DIM)
    print("Model set up!!!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    print("Optimizer functional!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training begins!!!")
    train_fewshot(model, loader, optimizer, device, config.EPOCHS)


if __name__ == "__main__":
    main()
