import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, patch_wise=True):
        """
        Args:
            temperature: ， softmax  sharpness
            patch_wise:  True， patch ；
        """
        super().__init__()
        self.temperature = temperature
        self.patch_wise = patch_wise

    def forward(self, feats):
        """
        feats: [B, C, H, W] latent  (z_e  z_q)
        return: scalar contrastive loss
        """
        B, C, H, W = feats.shape
        feats = F.normalize(feats, dim=1)  #  C ，

        if self.patch_wise:
            feats = feats.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
            feats = feats.reshape(B * H * W, C)
            labels = torch.arange(feats.shape[0], device=feats.device)
        else:
            feats = feats.mean(dim=[2, 3])  # [B, C]
            labels = torch.arange(B, device=feats.device)

        similarity_matrix = torch.matmul(feats, feats.T)  # [N, N]
        similarity_matrix = similarity_matrix / self.temperature

        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=feats.device)
        similarity_matrix.masked_fill_(mask, -1e9)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
