import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class GaborDirectionFrequencyExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
                                      
        self.conv_theta = nn.ModuleList([                                    
            nn.Conv2d(in_channels, 1, kernel_size=k, padding=k // 2) for k in [3, 5, 7]
        ])
        self.conv_freq = nn.ModuleList([                                  
            nn.Conv2d(in_channels, 1, kernel_size=k, padding=k // 2) for k in [3, 5, 7]
        ])

    def forward(self, x):
                           
        theta = sum([conv(x) for conv in self.conv_theta]) / len(self.conv_theta)
        freq = sum([conv(x) for conv in self.conv_freq]) / len(self.conv_freq)
        return theta, freq

                                                  
class TextureFlowConv(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super().__init__()
        self.num_heads = 6
        self.key_dim = 16

                                                      
        self.attn_qkv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, self.key_dim), nn.ReLU(),
                nn.Linear(self.key_dim, 1)
            ) for _ in range(self.num_heads)
        ])
                                
        self.gate_layer = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.node_proj = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(in_channels)

                               
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid()
        )

        self.gamma = 1.0          
        self.beta = 1.0           
        self.epsilon = 1e-6          
        self.sigma_theta = 1.0        
        self.sigma_f = 1.0        
        self.sigma_p = 1.0        
        self.lambda_spatial = 0.1          
        self.top_k = 32

        self.epoch = 0
        self.layer_idx = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, feat, theta, freq, uncertainty=None):
        B, C, H, W = feat.shape
        assert H * W <= 1024, f"TFGCNet graph too large: H={H}, W={W}, N={H * W}"
        x = feat.view(B, C, -1).permute(0, 2, 1)               
        theta = theta.view(B, -1, 1)
        freq = freq.view(B, -1, 1)

        out_list = []
        for b in range(B):
            feat_b = x[b]
            theta_b = theta[b]
            freq_b = freq[b]
            N = feat_b.shape[0]

            theta_pair = theta_b.unsqueeze(1) - theta_b.unsqueeze(0)
            freq_pair = freq_b.unsqueeze(1) - freq_b.unsqueeze(0)
            delta = torch.cat([theta_pair, freq_pair], dim=-1)
            theta_diff = torch.abs(theta_pair.squeeze(-1))          
            theta_diff = torch.remainder(theta_diff, torch.pi)
            theta_diff = torch.minimum(theta_diff, torch.pi - theta_diff)

            freq_diff = torch.abs(freq_pair.squeeze(-1))          

            ds = (
                    (theta_diff ** 2) / (self.sigma_theta ** 2) +
                    (freq_diff ** 2) / (self.sigma_f ** 2)
            )          

            yy, xx = torch.meshgrid(
                torch.arange(H, device=feat.device),
                torch.arange(W, device=feat.device),
                indexing='ij'
            )
            positions = torch.stack(
                [yy.reshape(-1), xx.reshape(-1)],
                dim=1
            ).float()

            dp = torch.cdist(positions, positions, p=2) ** 2
            dp = dp / (self.sigma_p ** 2)          

            if uncertainty is not None:
                uncertainty_b = uncertainty[b].reshape(-1)       

                if uncertainty_b.numel() != N:
                    raise RuntimeError(
                        f"TFGCNet uncertainty size mismatch: "
                        f"got {uncertainty_b.numel()}, expected {N}."
                    )

                b_v = 1.0 + self.gamma * uncertainty_b.view(1, N)          
            else:
                b_v = torch.ones(
                    1,
                    N,
                    device=feat.device,
                    dtype=feat.dtype
                )
            E_uv = torch.exp(
                -ds / (b_v ** 2 + 1e-6) -
                self.lambda_spatial * dp
            )          

            if uncertainty is not None:
                                                 
                confidence = 1 / (uncertainty_b + self.epsilon)
                confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)

                                       
                g_uv = torch.sigmoid(self.beta * confidence.unsqueeze(-1))
                E_tilde = E_uv * g_uv
            else:
                E_tilde = E_uv

                            
            dist = torch.norm(delta, dim=-1)
            gaussian_mask = torch.exp(-dist ** 2 / (2 * 0.2 ** 2))

                                 
            if E_tilde.dim() == 3 and E_tilde.size(-1) == 1:
                E_tilde = E_tilde.squeeze(-1)

                                       
            if gaussian_mask.dim() == 3 and gaussian_mask.size(-1) == 1:
                gaussian_mask = gaussian_mask.squeeze(-1)

                               
            assert E_tilde.shape == (N, N), f"E_tilde shape error: {E_tilde.shape}"
            assert gaussian_mask.shape == (N, N), f"gaussian_mask shape error: {gaussian_mask.shape}"

            affinity = E_tilde * gaussian_mask

            affinity = torch.clamp(affinity, min=1e-12)

            k = min(self.top_k, affinity.size(-1))

                                                 
            topk_values, topk_indices = torch.topk(
                affinity,
                k=k,
                dim=-1,
                largest=True,
                sorted=False
            )

            topk_mask = torch.zeros_like(affinity)
            topk_mask.scatter_(dim=-1, index=topk_indices, value=1.0)

            logits = torch.log(affinity)
            logits = logits.masked_fill(topk_mask == 0, -1e9)

            alpha = F.softmax(logits, dim=-1)

            alpha = alpha * topk_mask

            edge_gate = torch.sigmoid(self.gate_layer(delta).squeeze(-1))

            edge_weight = alpha * edge_gate

            edge_weight = edge_weight * topk_mask
            edge_weight = edge_weight / (edge_weight.sum(dim=-1, keepdim=True) + 1e-6)

            I = torch.eye(N, device=feat.device, dtype=edge_weight.dtype)
            A = edge_weight + I

            degree = A.sum(dim=-1)
            degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)

            A_norm = (
                    degree_inv_sqrt.unsqueeze(1) *
                    A *
                    degree_inv_sqrt.unsqueeze(0)
            )

            agg = torch.matmul(A_norm, feat_b)

            agg = self.node_proj(
                agg.unsqueeze(0).transpose(1, 2)
            ).transpose(1, 2).squeeze(0)

            agg = self.norm(agg)

            enhanced = agg.view(H, W, C).permute(2, 0, 1).contiguous()
            out_list.append(enhanced)

        out_tensor = torch.stack(out_list, dim=0)

        fusion_gate = self.fusion_gate(out_tensor)

        out_tensor = feat * (1.0 - fusion_gate) + out_tensor * fusion_gate

        return out_tensor

class TextureGraphEncoder(nn.Module):
    def __init__(self, in_channels=512, num_layers=2, return_intermediate=False):
        super().__init__()
        self.extractor = GaborDirectionFrequencyExtractor(in_channels)
        self.layers = nn.ModuleList([
            TextureFlowConv(in_channels) for _ in range(num_layers)
        ])
        for i, layer in enumerate(self.layers):
            layer.layer_idx = i
        self.return_intermediate = return_intermediate

    def set_epoch(self, epoch):
        for layer in self.layers:
            layer.set_epoch(epoch)

    def forward(self, x, uncertainty=None):
        theta, freq = self.extractor(x)
        intermediates = []
        for layer in self.layers:
            x = x + layer(x, theta, freq, uncertainty)          
            if self.return_intermediate:
                intermediates.append(x)
        return (x, intermediates) if self.return_intermediate else x
