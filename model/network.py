from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
import torch
from torchvision import models
import functools
import torch.nn as nn
import os
import torchvision.transforms as transforms
from model.texture_graph_module import TextureGraphEncoder
from model.csa_mspcnn import CSAMSPCNN
                                                                                                              
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def is_valid_module(m):
    return (
        isinstance(m, nn.Module)
        and hasattr(m, "forward")
        and callable(m.forward)
        and type(m.forward) is not nn.Module.forward
    )

def define_csa(init_type='orthogonal', gpu_ids=[]):
    net = CSAMSPCNN(iterations=8, eta=0.05)
    return init_net(net, init_type, gpu_ids)

def define_sgtfm(init_type='orthogonal', gpu_ids=[]):
    net = SGTFM()
    return init_net(net, init_type, gpu_ids)

def forward_module(m, x, features):
                   
    if not is_valid_module(m):
        print(f"[Skipped] Invalid module: {m.__class__.__name__}")
        return x

                            
    if isinstance(m, nn.ModuleList):
        for sub_m in m:
            if not is_valid_module(sub_m):
                print(f"[Skipped] Invalid submodule: {sub_m.__class__.__name__}")
                continue
            x = forward_module(sub_m, x, features)
        return x

          
    try:
        x = m(x)
        features.append(x)
    except Exception as e:
        raise RuntimeError(
            f"SUPNet VQGAN encoder module {m.__class__.__name__} failed: {e}"
        )
    return x

                                                                


import torch.serialization
from pytorch_lightning.callbacks import ModelCheckpoint

class SUPNet(nn.Module):
    """Structural Uncertainty Prediction Network for mural structure completion"""

    def __init__(self, config_path, ckpt_path):
        super().__init__()

        self.vqgan_encoder = self._load_vqgan_encoder(config_path, ckpt_path)
        self.vqgan_decoder = self._load_vqgan_decoder(config_path, ckpt_path)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),              
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Softplus()
        )
        self.structure_predictor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),              
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def _load_vqgan_encoder(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)
        torch.serialization.add_safe_globals([ModelCheckpoint])
        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict']
        model_state = model.state_dict()

        for key in ['encoder.conv_in.weight', 'encoder.conv_in.bias']:
            if key in state_dict and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    print(f"Delete mismatched key: {key}, ckpt={state_dict[key].shape}, model={model_state[key].shape}")
                    del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model.encoder


    def _load_vqgan_decoder(self, config_path, ckpt_path):
        """VQGAN_decoder"""
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)

        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict']

        for key in ['encoder.conv_in.weight', 'encoder.conv_in.bias']:
            if key in state_dict:
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model.decoder

    def compute_saliency_map(self, image):
        image_rgb = image[:, :3, :, :]

        if image_rgb.min() < 0:
            image_rgb = (image_rgb + 1.0) / 2.0

        image_rgb = torch.clamp(image_rgb, 0.0, 1.0)

        image_gray = (
                0.299 * image_rgb[:, 0:1, :, :] +
                0.587 * image_rgb[:, 1:2, :, :] +
                0.114 * image_rgb[:, 2:3, :, :]
        )


        grad_x = F.conv2d(image_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(image_gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)


        var_map = F.avg_pool2d(image_gray ** 2, 3, stride=1, padding=1) - F.avg_pool2d(image_gray, 3, stride=1,
                                                                                       padding=1) ** 2


        high_freq = torch.abs(image_gray - F.avg_pool2d(image_gray, 3, stride=1, padding=1))


        saliency = 0.4 * grad_mag + 0.3 * var_map + 0.3 * high_freq
        return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    def generate_structure_mask(self, saliency, kappa_t):
        flat_saliency = saliency.view(-1)
        threshold = torch.quantile(flat_saliency, 1 - kappa_t)
        mask = (saliency >= threshold).float()
        return mask

    def compute_nll_loss(self, S_rec, S_gt, delta, mask=None):
        if S_gt is None:
            return None

              
        if S_gt.shape[2:] != S_rec.shape[2:]:
            S_gt = F.interpolate(
                S_gt,
                size=S_rec.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        if delta.shape[2:] != S_rec.shape[2:]:
            delta = F.interpolate(
                delta,
                size=S_rec.shape[2:],
                mode='bilinear',
                align_corners=False
            )

              
        if S_gt.size(1) != S_rec.size(1):
            if S_gt.size(1) == 1 and S_rec.size(1) == 3:
                S_gt = S_gt.repeat(1, 3, 1, 1)
            elif S_gt.size(1) == 3 and S_rec.size(1) == 1:
                S_gt = S_gt.mean(dim=1, keepdim=True)
            else:
                                             
                S_gt = S_gt.mean(dim=1, keepdim=True)
                S_rec = S_rec.mean(dim=1, keepdim=True)

                                  
        delta = torch.clamp(delta, min=1e-3, max=10.0)

        sq_error = (S_rec - S_gt) ** 2
        sq_error = sq_error.mean(dim=1, keepdim=True)

        nll = sq_error / (2.0 * delta) + 0.5 * torch.log(delta)

                         
        if mask is not None:
            if mask.shape[2:] != nll.shape[2:]:
                mask = F.interpolate(mask, size=nll.shape[2:], mode='nearest')

            if mask.size(1) != 1:
                mask = mask[:, :1, :, :]

            valid = mask.float()
            loss = (nll * valid).sum() / (valid.sum() + 1e-6)
        else:
            loss = nll.mean()

        return loss

    def build_structure_gt(self, image):
        image_rgb = image[:, :3, :, :]

                                 
        if image_rgb.min() < 0:
            image_rgb = (image_rgb + 1.0) / 2.0

        image_rgb = torch.clamp(image_rgb, 0.0, 1.0)

        gray = (
                0.299 * image_rgb[:, 0:1, :, :] +
                0.587 * image_rgb[:, 1:2, :, :] +
                0.114 * image_rgb[:, 2:3, :, :]
        )

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)

        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        edge_min = edge.amin(dim=(2, 3), keepdim=True)
        edge_max = edge.amax(dim=(2, 3), keepdim=True)
        edge = (edge - edge_min) / (edge_max - edge_min + 1e-6)

        return edge

    def forward(self,image,mask=None,iteration=0,total_iterations=100000,S_gt=None,use_mask_loss=True):

        image_rgb = image[:, :3, :, :]

        if mask is not None and mask.size(1) != 1:
            mask_for_loss = mask[:, :1, :, :]
        else:
            mask_for_loss = mask

                                          
        if hasattr(self.vqgan_encoder, "conv_in"):
            encoder_in_channels = self.vqgan_encoder.conv_in.in_channels
        else:
            encoder_in_channels = image.size(1)

        if encoder_in_channels == 3:
            image_for_encoder = image_rgb

        elif encoder_in_channels == 4:
            if mask_for_loss is None:
                raise RuntimeError("SUPNet encoder expects 4 channels, but mask is None.")
            image_for_encoder = torch.cat([image_rgb, mask_for_loss], dim=1)

        elif encoder_in_channels == 6:
            if mask_for_loss is None:
                raise RuntimeError("SUPNet encoder expects 6 channels, but mask is None.")
            mask3_for_encoder = mask_for_loss.repeat(1, 3, 1, 1)
            image_for_encoder = torch.cat([image_rgb, mask3_for_encoder], dim=1)

        else:
            raise RuntimeError(
                f"Unsupported VQGAN encoder input channels: {encoder_in_channels}"
            )

        features = []
        x = image_for_encoder

        for name, module in self.vqgan_encoder.named_children():
            x = forward_module(module, x, features)
            if len(features) >= 6:
                break

        if len(features) == 0:
            raise RuntimeError("SUPNet: VQGAN encoder did not return any feature.")

        F_s = features[-1]
        saliency = self.compute_saliency_map(image_rgb)

        ratio = float(iteration) / max(float(total_iterations), 1.0)
        ratio = max(0.0, min(1.0, ratio))

        kappa_t = 0.2 + (0.7 - 0.2) * ratio

        M_sp = self.generate_structure_mask(saliency, kappa_t)

        if M_sp.shape[2:] != F_s.shape[2:]:
            M_sp = F.interpolate(
                M_sp,
                size=F_s.shape[2:],
                mode='nearest'
            )

        F_s_masked = (1.0 - M_sp) * F_s
        delta = self.uncertainty_head(F_s)
        delta = torch.clamp(delta, min=1e-3, max=10.0)

        F_s_miss = self.structure_predictor(F_s_masked)

        F_s_filled = (1.0 - M_sp) * F_s + M_sp * F_s_miss

        S_rec = self.vqgan_decoder(F_s_filled)

        if S_gt is None and self.training:
            S_gt = self.build_structure_gt(image_for_encoder)

        loss_struct = None
        if S_gt is not None:
            loss_mask = mask_for_loss if use_mask_loss else None
            loss_struct = self.compute_nll_loss(
                S_rec=S_rec,
                S_gt=S_gt,
                delta=delta,
                mask=loss_mask
            )
                                                                      
        if F_s_filled.size(1) >= 512:
            distribution = [torch.split(F_s_filled, 256, dim=1)[:2]]
        elif F_s_filled.size(1) >= 256:
                                                          
            q_mu = F_s_filled
            q_sigma = torch.ones_like(F_s_filled)
            distribution = [[q_mu, q_sigma]]
        else:
            q_mu = F_s_filled
            q_sigma = torch.ones_like(F_s_filled)
            distribution = [[q_mu, q_sigma]]

        self.uncertainty_map = delta
        self.structure_rec = S_rec
        self.structure_mask = M_sp
        self.structure_loss = loss_struct
        self.structure_feature = F_s_filled

        return {
            "distribution": distribution,
            "features": features + [F_s_filled],
            "F_s": F_s,
            "F_s_masked": F_s_masked,
            "F_s_miss": F_s_miss,
            "F_s_filled": F_s_filled,
            "S_rec": S_rec,
            "delta": delta,
            "M_sp": M_sp,
            "loss_struct": loss_struct,
        }

def define_es(config_path=None, ckpt_path=None, init_type='normal', gpu_ids=[]):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if config_path is None:
        config_path = os.path.join(project_root, 'ckpt', 'dunhuang_vqgan.yaml')

    if ckpt_path is None:
        ckpt_path = os.path.join(project_root, 'ckpt', 'vq.ckpt')

    net = SUPNet(config_path, ckpt_path)
    return init_net(net, init_type, gpu_ids)


def define_et(init_type='orthogonal', gpu_ids=[], num_layers=2, return_intermediate=False):
    graph = TextureGraphEncoder(
        in_channels=512,
        num_layers=num_layers,
        return_intermediate=return_intermediate
    )
    net = Encoder_T_TextureEnhanced(graph)
    return init_net(net, init_type, gpu_ids)



def define_de(init_type='orthogonal', gpu_ids=[]):               

    net = Decoder()

    return init_net(net, init_type, gpu_ids)


def define_dis_g(init_type='orthogonal', gpu_ids=[]):

    net = GlobalDiscriminator()                             

    return init_net(net, init_type, gpu_ids)


def define_attn(in_channels_s=128, in_channels_t=512, out_channels=128,
                init_type='orthogonal', gpu_ids=[]):
    net = Cross_Attention(in_channels_s, in_channels_t, out_channels)

    return init_net(net, init_type, gpu_ids)

def define_fuse_s(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_s()                        

    return init_net(net, init_type, gpu_ids)

def define_fuse_t(init_type='orthogonal', gpu_ids=[]):

    net = Trans_conv_t()                        

    return init_net(net, init_type, gpu_ids)

def define_G2(init_type='orthogonal', gpu_ids=[]):

    net = refine_G2()                    

    return init_net(net, init_type, gpu_ids)



class Encoder_T_TextureEnhanced(nn.Module):
    def __init__(self, texture_graph_module):
        super().__init__()
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gate_nonlinearity = nn.Sigmoid()

              
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}

                  
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_gate))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)

                  
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

                  
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

                  
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 256, **kwargs))
        self.shortcut4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

                  
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_gate))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

                    
        self.texture_graph_module = texture_graph_module

    def forward(self, x, uncertainty=None):
        feature = []

        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)

        x = self.nonlinearity(self.pool(self.model2(x)) + self.pool(self.shortcut2(x))) * self.gate2(x)
        feature.append(x)

        x = self.nonlinearity(self.pool(self.model3(x)) + self.pool(self.shortcut3(x))) * self.gate3(x)
        feature.append(x)

        x = self.nonlinearity(self.pool(self.model4(x)) + self.pool(self.shortcut4(x))) * self.gate4(x)
        feature.append(x)

        x = self.nonlinearity(self.pool(self.model5(x)) + self.pool(self.shortcut5(x))) * self.gate5(x)
        feature.append(x)

        if uncertainty is not None and uncertainty.shape[2:] != x.shape[2:]:
            uncertainty = F.interpolate(uncertainty, size=x.shape[2:], mode='bilinear', align_corners=False)

        enhanced_texture = self.texture_graph_module(x, uncertainty)
        feature.append(enhanced_texture)

        q_mu, q_std = torch.split(enhanced_texture, 256, dim=1)
        q_std = F.softplus(q_std) + 1e-6

        return [[q_mu, q_std]], feature

class SGTFM(nn.Module):
    """
    Structure-Guided Texture Fusion Module.
    """

    def __init__(
        self,
        s_channels=(128, 128, 128),
        t_channels=(128, 256, 512),
        out_channels=512
    ):
        super().__init__()

        self.out_channels = out_channels
        self.alpha_delta = nn.Parameter(torch.ones(1))

                                  
        self.s_proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in s_channels
        ])

                                                                
        self.t_proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in t_channels
        ])

        self.q_proj = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            for _ in range(3)
        ])

        self.k_proj = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            for _ in range(3)
        ])

        self.v_proj = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
            for _ in range(3)
        ])

                 
        self.scale_logits = nn.Parameter(torch.zeros(3))

                                
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

                                
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def cross_attend(self, s, t, delta, idx):
        B, C, H, W = t.shape

        if s.shape[2:] != (H, W):
            s = F.interpolate(
                s,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        if delta.shape[2:] != (H, W):
            delta = F.interpolate(
                delta,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        q = self.q_proj[idx](t).flatten(2).transpose(1, 2)               
        k = self.k_proj[idx](s).flatten(2)                               
        v = self.v_proj[idx](s).flatten(2).transpose(1, 2)               

                
        b_delta = self.alpha_delta.abs() * (1.0 + delta)
        omega = torch.exp(-1.0 / (b_delta ** 2 + 1e-6))                    
        omega = omega.flatten(2).transpose(1, 2)                         

        attn = torch.bmm(q, k) / (C ** 0.5)                               
        attn = attn * omega                                                  
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)                                         
        out = out.transpose(1, 2).view(B, C, H, W)                         

        return out

    def forward(self, s_features, t_features, delta):

        if len(s_features) < 1:
            raise RuntimeError("SGTFM: s_features is empty.")

        if len(t_features) < 4:
            raise RuntimeError(
                f"SGTFM: t_features length is too short, got {len(t_features)}."
            )

        s_last = s_features[-1]
        s_list = [s_last, s_last, s_last]

        t_list = [t_features[2], t_features[3], t_features[-1]]

        target_size = t_list[-1].shape[2:]
        fused = []

        for i in range(3):
            s = s_list[i]
            t = t_list[i]
            s = self.s_proj[i](s)
            t = self.t_proj[i](t)

                               
            if s.shape[2:] != target_size:
                s = F.interpolate(
                    s,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            if t.shape[2:] != target_size:
                t = F.interpolate(
                    t,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            fused_i = self.cross_attend(s, t, delta, i)
            fused.append(fused_i)

        weights = F.softmax(self.scale_logits, dim=0)

        f = (
            weights[0] * fused[0] +
            weights[1] * fused[1] +
            weights[2] * fused[2]
        )
        ca = self.channel_attn(f)
        f = f * ca + f

        avg_map = torch.mean(f, dim=1, keepdim=True)
        max_map, _ = torch.max(f, dim=1, keepdim=True)

        sa = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
        f = f * sa + f

        return self.out_proj(f)

class Decoder(nn.Module):
    def __init__(self):                
        super(Decoder, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        kwargs_fuse_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.gate_nonlinearity = nn.Sigmoid()
        self.conv1 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model1 = nn.Sequential(self.nonlinearity, self.conv1, self.nonlinearity, self.conv2)
        self.gateconv1 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)
        self.conv3 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.shortcut2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.model2 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv3, self.norm(512), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_short))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)
        self.conv5 = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_short))
        self.shortcut3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv5, self.norm(256), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.ConvTranspose2d(512, 128, **kwargs_short))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

        self.conv_out1 = SpectralNorm(nn.Conv2d(128, 3, **kwargs_out))
        self.model_out1 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out1, nn.Tanh())

        self.conv7 = SpectralNorm(nn.Conv2d(259, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.shortcut4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.model4 = nn.Sequential(self.norm(259), self.nonlinearity, self.conv7, self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.ConvTranspose2d(259, 64, **kwargs_short))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

        self.conv_out2 = SpectralNorm(nn.Conv2d(64, 3, **kwargs_out))
        self.model_out2 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out2, nn.Tanh())

        self.conv9 = SpectralNorm(nn.Conv2d(131, 32, **kwargs))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.shortcut5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.model5 = nn.Sequential(self.norm(131), self.nonlinearity, self.conv9, self.norm(32), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.ConvTranspose2d(131, 32, **kwargs_short))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

        self.conv_out3 = SpectralNorm(nn.Conv2d(32, 3, **kwargs_out))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out3, nn.Tanh())

        self.conv11 = SpectralNorm(nn.Conv2d(67, 16, **kwargs))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(16, 16, **kwargs_short))
        self.shortcut6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.model6 = nn.Sequential(self.norm(67), self.nonlinearity, self.conv11, self.norm(16), self.nonlinearity, self.conv12)
        self.gateconv6 = SpectralNorm(nn.ConvTranspose2d(67, 16, **kwargs_short))
        self.gate6 = nn.Sequential(self.gateconv6, self.gate_nonlinearity)
             
        self.conv_out4 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out4 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out4, nn.Tanh())

        self.conv13 = SpectralNorm(nn.Conv2d(19, 16, **kwargs))
        self.conv14 = SpectralNorm(nn.Conv2d(16, 16, **kwargs))
        self.shortcut7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.model7 = nn.Sequential(self.norm(19), self.nonlinearity, self.conv13, self.norm(16), self.nonlinearity, self.conv14)
        self.gateconv7 = SpectralNorm(nn.ConvTranspose2d(19, 16, **kwargs))
        self.gate7 = nn.Sequential(self.gateconv7, self.gate_nonlinearity)

        self.conv_out5 = SpectralNorm(nn.Conv2d(16, 3, **kwargs_out))
        self.model_out5 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out5, nn.Tanh())

        self.fuse_conv1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv2 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv3 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down1 = nn.Sequential(self.fuse_conv1, self.nonlinearity, self.fuse_conv2, self.nonlinearity, self.fuse_conv3, self.nonlinearity)

        self.fuse_conv4 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_conv4_1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down2 = nn.Sequential(self.fuse_conv4, self.nonlinearity, self.fuse_conv4_1, self.nonlinearity)

        self.fuse_conv5 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_fuse_down))
        self.fuse_down3 = nn.Sequential(self.fuse_conv5, self.nonlinearity)

        self.fuse_conv6 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_up1 = nn.Sequential(self.fuse_conv6, self.norm(128), self.nonlinearity)

        self.fuse_conv8 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv10 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv11 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_short))
        self.fuse_up2 = nn.Sequential(self.fuse_conv8, self.norm(128), self.nonlinearity, self.fuse_conv10, self.norm(64), self.nonlinearity,
                                      self.fuse_conv11, self.norm(64), self.nonlinearity)

        self.fuse_conv12 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.fuse_conv13 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.fuse_conv14 = SpectralNorm(nn.Conv2d(64, 32, **kwargs))
        self.fuse_conv15 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_conv16 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.fuse_conv17 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_short))
        self.fuse_up3 = nn.Sequential(self.fuse_conv12, self.norm(128), self.nonlinearity, self.fuse_conv13, self.norm(64), self.nonlinearity,
                                      self.fuse_conv14, self.norm(32), self.nonlinearity, self.fuse_conv15, self.norm(32), self.nonlinearity,
                                      self.fuse_conv16, self.norm(32), self.nonlinearity, self.fuse_conv17, self.norm(32), self.nonlinearity,)

    def forward(self, x, fuse_s, fuse_t):
        results = []

        s_1 = self.fuse_down1(fuse_s)
        s_2 = self.fuse_down2(fuse_s)
        s_3 = self.fuse_down3(fuse_s)

        t_1 = self.fuse_up1(fuse_t)
        t_2 = self.fuse_up2(fuse_t)
        t_3 = self.fuse_up3(fuse_t)

        out = x
                
        if out.shape[2:] != s_1.shape[2:]:
                                                                
            s_1 = F.interpolate(s_1, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, s_1], dim=1)
        out = self.nonlinearity(self.model1(out) + self.shortcut1(out)) * self.gate1(out)

        if out.shape[2:] != s_2.shape[2:]:
                                                                
            s_2 = F.interpolate(s_2, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, s_2], 1)
        out = self.nonlinearity(self.model2(out) + self.shortcut2(out)) * self.gate2(out)

        if out.shape[2:] != s_3.shape[2:]:
                                                                
            s_3 = F.interpolate(s_3, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, s_3], 1)
        out = self.nonlinearity(self.model3(out) + self.shortcut3(out)) * self.gate3(out)

        output = self.model_out1(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)

                
        if out.shape[2:] != t_1.shape[2:]:
                                                                
            t_1 = F.interpolate(t_1, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, t_1], 1)
        out = self.nonlinearity(self.model4(out) + self.shortcut4(out)) * self.gate4(out)

        output = self.model_out2(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)

                
        if out.shape[2:] != t_2.shape[2:]:
                                                                
            t_2 = F.interpolate(t_2, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, t_2], 1)
        out = self.nonlinearity(self.model5(out) + self.shortcut5(out)) * self.gate5(out)

        output = self.model_out3(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)

                
        if out.shape[2:] != t_3.shape[2:]:
                                                                
            t_3 = F.interpolate(t_3, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out, t_3], 1)
        out = self.nonlinearity(self.model6(out) + self.shortcut6(out)) * self.gate6(out)

        output = self.model_out4(out)
        results.append(output)
        out = torch.cat([out, output], dim=1)

        out = self.nonlinearity(self.model7(out) + self.shortcut7(out)) * self.gate7(out)
        output = self.model_out5(out)
        results.append(output)

        return results


class GlobalDiscriminator(nn.Module):
    def __init__(self):                              
        super(GlobalDiscriminator, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

                  
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)

                  
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.bypass2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.shortcut2 = nn.Sequential(self.bypass2)

                  
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.bypass3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.shortcut3 = nn.Sequential(self.bypass3)

                  
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.shortcut4 = nn.Sequential(self.bypass4)

                  
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.shortcut5 = nn.Sequential(self.bypass5)

                  
        self.conv11 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model6 = nn.Sequential(self.nonlinearity, self.conv11, self.nonlinearity, self.conv12)
        self.shortcut6 = nn.Sequential(self.bypass6)

                
        self.concat = SpectralNorm(nn.Conv2d(128, 1, 3))

    def forward(self, x):                                                          
              
        x = self.model1(x) + self.shortcut1(x)
              
                                        
                                                                      
                                                         
        x = self.pool(self.model2(x)) + self.pool(self.shortcut2(x))
        x = self.pool(self.model3(x)) + self.pool(self.shortcut3(x))
        out = self.pool(self.model4(x)) + self.pool(self.shortcut4(x))
        out = self.pool(self.model5(out)) + self.pool(self.shortcut5(out))
        out = self.pool(self.model6(out)) + self.pool(self.shortcut6(out))
                                                        
        out = self.concat(self.nonlinearity(out))

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

class Cross_Attention(nn.Module):
    def __init__(self, in_channels_s, in_channels_t, out_channels=128):
        super(Cross_Attention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels_t, out_channels // 2, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels_s, out_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels_s, 512, kernel_size=1)                   

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s, x_t):
        B, _, H, W = x_t.shape

                                      
        if x_s.shape[2:] != x_t.shape[2:]:
            x_s = F.interpolate(x_s, size=(H, W), mode='bilinear', align_corners=False)

                                                                     

                               
        proj_query = self.query_conv(x_t).view(B, -1, H * W).permute(0, 2, 1)
        proj_key   = self.key_conv(x_s).view(B, -1, H * W).permute(0, 2, 1)
        proj_value = self.value_conv(x_s).view(B, -1, H * W)

                                                                                                  

        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))               
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                 
        out = out.view(B, -1, H, W)
        return out

class Trans_conv_s(nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super(Trans_conv_s, self).__init__()
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

                      
        self.shared_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            self.norm(out_channels),
            self.nonlinearity
        )

        self.down = nn.Sequential(
            SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=1)),
            self.norm(out_channels),
            self.nonlinearity
        )

    def forward(self, features):
        base_size = features[0].shape[2:]
        conv_outs = []

        for f in features:
            if f.shape[2:] != base_size:
                f = F.interpolate(f, size=base_size, mode='bilinear', align_corners=False)
            conv_outs.append(self.shared_conv(f))

        x = torch.cat(conv_outs, dim=1)                      

        x = self.down(x)
        return x


class Trans_conv_t(nn.Module):
    def __init__(self):                         
        super(Trans_conv_t, self).__init__()
        kwargs_short = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)


        self.conv1 = SpectralNorm(nn.Conv2d(32, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model1 = nn.Sequential(self.conv1, self.norm(128), self.nonlinearity, self.conv2, self.norm(256), self.nonlinearity)

        self.conv3 = SpectralNorm(nn.Conv2d(64, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
        self.model2 = nn.Sequential(self.conv3, self.norm(256), self.nonlinearity)

        self.conv4 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.model3 = nn.Sequential(self.conv4, self.norm(256), self.nonlinearity)

        self.conv5 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv6 = SpectralNorm(nn.ConvTranspose2d(256, 256,**kwargs_short))
        self.model4 = nn.Sequential(self.conv5, self.norm(256), self.nonlinearity, self.conv6, self.norm(256), self.nonlinearity)

        self.conv7 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv8 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv9 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv10 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model5 = nn.Sequential(self.conv7, self.norm(256), self.nonlinearity, self.conv8, self.norm(256),self.nonlinearity,
                                    self.conv9, self.norm(256), self.nonlinearity, self.conv10, self.norm(256),self.nonlinearity)

        self.conv11 = SpectralNorm(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv12 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv13 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv14 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.conv15 = SpectralNorm(nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.conv16 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_short))
        self.model6 = nn.Sequential(self.conv11, self.norm(256), self.nonlinearity, self.conv12, self.norm(256),self.nonlinearity,
                                    self.conv13, self.norm(256), self.nonlinearity, self.conv14, self.norm(256),self.nonlinearity,
                                    self.conv15, self.norm(256), self.nonlinearity, self.conv16, self.norm(256),self.nonlinearity)

        self.conv17 = SpectralNorm(nn.Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1)))
        self.down = nn.Sequential(self.conv17, self.norm(256), self.nonlinearity)
    def forward(self, feature1):             
        x1 = self.model1(feature1[0])
        x2 = self.model2(feature1[1])
        x3 = self.model3(feature1[2])
        x4 = self.model4(feature1[3])
        x5 = self.model5(feature1[4])
        x6 = self.model6(feature1[5])
                                  
        if x6.shape[2:] != x5.shape[2:]:
            x6 = F.interpolate(x6, size=x5.shape[2:], mode='bilinear', align_corners=False)
        x_fuse = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        x = self.down(x_fuse)

        return x


class refine_G2(nn.Module):
    def __init__(self):                     
        super(refine_G2, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_3 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_5 = {'kernel_size': 5, 'stride': 1, 'padding': 2}
        kwargs_7 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        kwargs_up = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

                  
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))                                                                                                          
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.shortcut1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))               
        self.model1 = nn.Sequential(self.conv1, self.norm(32), self.nonlinearity, self.conv2)                                   
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))                               
        self.gate1 = nn.Sequential(self.gateconv1, self.gate_nonlinearity)                                                  

                  
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.model2 = nn.Sequential(self.norm(32), self.nonlinearity, self.conv3, self.norm(32), self.nonlinearity, self.conv4)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_down))
        self.gate2 = nn.Sequential(self.gateconv2, self.gate_nonlinearity)

                  
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.model3 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv5, self.norm(64), self.nonlinearity, self.conv6)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.gate3 = nn.Sequential(self.gateconv3, self.gate_nonlinearity)

                  
        self.conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.shortcut4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv7, self.norm(64), self.nonlinearity, self.conv8)
        self.gateconv4 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_down))
        self.gate4 = nn.Sequential(self.gateconv4, self.gate_nonlinearity)

                  
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.model5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv9, self.norm(128), self.nonlinearity, self.conv10)
        self.gateconv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.gate5 = nn.Sequential(self.gateconv5, self.gate_nonlinearity)

                   
        self.conv_3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_3))
        self.multi_3 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_3)
        self.conv_5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_5))
        self.multi_5 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_5)
        self.conv_7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_7))
        self.multi_7 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv_7)

                  
        self.de_conv1 = SpectralNorm(nn.Conv2d(384, 384, **kwargs))
        self.de_conv2 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_shortcut1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_model1 = nn.Sequential(self.norm(384), self.nonlinearity, self.de_conv1, self.norm(384), self.nonlinearity, self.de_conv2)
        self.de_gateconv1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.de_gate1 = nn.Sequential(self.de_gateconv1, self.gate_nonlinearity)

                  
        self.de_conv3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv4 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_shortcut2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_model2 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv3, self.norm(128), self.nonlinearity,
                                    self.de_conv4)
        self.de_gateconv2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs))
        self.de_gate2 = nn.Sequential(self.de_gateconv2, self.gate_nonlinearity)

        self.attn1 = Auto_Attn(128)

                  
        self.de_conv5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.de_conv6 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_shortcut3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_model3 = nn.Sequential(self.norm(128), self.nonlinearity, self.de_conv5, self.norm(128), self.nonlinearity,
                                    self.de_conv6)
        self.de_gateconv3 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_up))
        self.de_gate3 = nn.Sequential(self.de_gateconv3, self.gate_nonlinearity)

                  
        self.de_conv7 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv8 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_shortcut4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_model4 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv7, self.norm(64), self.nonlinearity,
                                    self.de_conv8)
        self.de_gateconv4 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs))
        self.de_gate4 = nn.Sequential(self.de_gateconv4, self.gate_nonlinearity)

        self.attn2 = Auto_Attn(64)

                  
        self.de_conv9 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.de_conv10 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_shortcut5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_model5 = nn.Sequential(self.norm(64), self.nonlinearity, self.de_conv9, self.norm(64), self.nonlinearity,
                                    self.de_conv10)
        self.de_gateconv5 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_up))
        self.de_gate5 = nn.Sequential(self.de_gateconv5, self.gate_nonlinearity)

        self.out = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.model_out = nn.Sequential(self.nonlinearity, self.out, nn.Tanh())

    def forward(self, x, mask):                   
        x = torch.cat([x, mask], dim=1)                                                              
        feature = []                 
                                                                                
                                              
                                                                                                          
        x = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        feature.append(x)
        x = self.nonlinearity(self.model2(x) + self.shortcut2(x)) * self.gate2(x)
        feature.append(x)
        x = self.nonlinearity(self.model3(x) + self.shortcut3(x)) * self.gate3(x)
        feature.append(x)
        x = self.nonlinearity(self.model4(x) + self.shortcut4(x)) * self.gate4(x)
        feature.append(x)
        x = self.nonlinearity(self.model5(x) + self.shortcut5(x)) * self.gate5(x)
        feature.append(x)

                                                                                 
        multi1 = self.multi_3(x)
        multi2 = self.multi_5(x)
        multi3 = self.multi_7(x)
        fuse = torch.cat([multi1, multi2, multi3], 1)

             
                                                                                                           
                                                                                                                 
        x = self.nonlinearity(self.de_model1(fuse) + self.de_shortcut1(fuse)) * self.de_gate1(fuse)
        x = self.nonlinearity(self.de_model2(x) + self.de_shortcut2(x)) * self.de_gate2(x)
        x = self.attn1(x, feature[3], mask)
        x = self.nonlinearity(self.de_model3(x) + self.de_shortcut3(x)) * self.de_gate3(x)
        x = self.nonlinearity(self.de_model4(x) + self.de_shortcut4(x)) * self.de_gate4(x)
        x = self.attn2(x, feature[1], mask)
        x = self.nonlinearity(self.de_model5(x) + self.de_shortcut5(x)) * self.de_gate5(x)
        x = self.model_out(x)


        return x


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):                
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

                         
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):                 
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]