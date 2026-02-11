import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True,freeze_backbone=True):
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Remove classifier head
        self.vit.heads = nn.Identity()

        # MLP Head (Projection Head)
        self.mlp_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
        
        #freeze backbone
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat = self.vit(x)          # (B, 768)
        emb = self.mlp_head(feat)   # (B, embed_dim)
        return F.normalize(emb, dim=1)



