                                              
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraPatchContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_radius=1, neg_radius=4, patch_num_samples=128):
        """
        pos_radius: patch （ 1  3x3 ）
        neg_radius: patch （ 4  ≥4 ）
        patch_num_samples:  anchor patch 
        """
        super().__init__()
        self.temperature = temperature
        self.pos_radius = pos_radius
        self.neg_radius = neg_radius
        self.patch_num_samples = patch_num_samples

    def forward(self, z):
        B, C, H, W = z.shape
        device = z.device

                       
        z = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        loss = 0.0
        total = 0

        for b in range(B):
            z_b = z[b]                           
            z_b = F.normalize(z_b, dim=-1)

                   
            coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1)
            coords = coords.view(-1, 2).to(device)            

                           
            sample_indices = torch.randperm(H * W, device=device)[:self.patch_num_samples]
            anchor_coords = coords[sample_indices]
            anchor_feats = z_b[sample_indices]          

            for i, anchor_feat in enumerate(anchor_feats):
                anchor_xy = anchor_coords[i]

                  
                                                                        
                dists = torch.norm(coords.float() - anchor_xy.float(), dim=1)
                pos_mask = (dists <= self.pos_radius) & (dists > 0)
                neg_mask = dists >= self.neg_radius

                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue    

                pos_feats = z_b[pos_mask]          
                neg_feats = z_b[neg_mask]          

                  
                pos_sim = torch.matmul(anchor_feat.unsqueeze(0), pos_feats.T) / self.temperature
                pos_sim = pos_sim.mean()

                  
                neg_sim = torch.matmul(anchor_feat.unsqueeze(0), neg_feats.T) / self.temperature
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim.squeeze(0)], dim=0)
                labels = torch.zeros(logits.size(0), device=device)
                labels[0] = 1    

                                              
                contrastive_loss = -F.log_softmax(logits, dim=0)[0]
                loss += contrastive_loss
                total += 1

        if total > 0:
            return loss / total
        else:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
