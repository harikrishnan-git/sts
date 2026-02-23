import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.vit = ViT(
            in_channels=1,        # CT / MRI / PET
            img_size=(224, 224),
            patch_size=(16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type="conv",
            classification=False,
            spatial_dims=2
        )

        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x):
      tokens, _ = self.vit(x)          # [B, 196, 768]
      cls = tokens[:, 0, :]            # CLS token
      emb = self.proj(cls)             # [B, 256]
      return F.normalize(emb, dim=1)

