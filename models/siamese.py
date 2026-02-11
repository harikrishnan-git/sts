import torch.nn as nn
from models.vit_encoder import ViTEncoder


class SiameseViT(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.encoder = ViTEncoder(embed_dim)

    def encode(self, x):
        return self.encoder(x)
