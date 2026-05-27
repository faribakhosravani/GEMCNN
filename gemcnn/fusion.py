import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """
    Concatenation → BatchNorm → Linear → log-softmax.
    Receives pre-flattened embeddings from all encoders.
    """
    def __init__(self, fusion_dim: int, num_classes: int):
        super().__init__()
        self.bn   = nn.BatchNorm1d(fusion_dim)
        self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: arbitrary number of (B, D_i) tensors
        fused  = torch.cat(embeddings, dim=1)   # (B, fusion_dim)
        fused  = self.bn(fused)
        logits = self.head(fused)               # (B, num_classes)
        return F.log_softmax(logits, dim=-1)
