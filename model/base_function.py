import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .external_function import SpectralNorm
from util import task

def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            init.orthogonal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


                                                                                          
def init_net(net, init_type='normal', gpu_ids=[]):
    """print the network structure and initial the network"""
    num_params = 0
    for param in net.parameters():                                                                           
        num_params += param.numel()
    print(net)                       
    print('total number of parameters: %.3f M' % (num_params / 1e6))                                    

    if len(gpu_ids) > 0:                                                                          
        assert(torch.cuda.is_available())
        net.cuda()                               
        net = torch.nn.DataParallel(net, gpu_ids)                                                                                            
    init_weights(net, init_type)                                             
    return net

def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

                         
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
                       
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=(1,1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.nonlinearity = nn.LeakyReLU(0.1)

        self.conv1 = SpectralNorm(nn.Conv2d(input_nc * 2, input_nc, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(input_nc, input_nc, **kwargs))
        self.shortcut = SpectralNorm(nn.Conv2d(input_nc * 2, input_nc, kernel_size=(1, 1), stride=(1, 1)))
        self.model = nn.Sequential(self.nonlinearity, self.conv1, self.nonlinearity, self.conv2)

    def forward(self, x, pre, mask):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
                           
        proj_query = self.query_conv(x).view(B, -1, W * H)              
        proj_key = proj_query               

                                            
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)                                          
        attention = self.softmax(energy)                       
        proj_value = x.view(B, -1, W * H)                       

                
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x                                  

        if type(pre) != type(None):                                            
                                                                                        
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)       
            scale_mask = task.scale_img(mask, size=[pre.size(2), pre.size(3)])                                                           
            M = scale_mask.chunk(3, dim=1)[0]                           
                        
            context_flow = self.alpha * (M) * context_flow + (1 - M) * pre                                                                     
                      
            input = torch.cat([out, context_flow], dim=1)                                           
            out = self.model(input) + self.shortcut(input)          

        return out


def style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value


def perceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value