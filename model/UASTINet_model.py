import torch
from .base_model import BaseModel
import torch.nn.functional as F

from . import network, base_function, external_function
from util import task
import itertools

class UASTINet(BaseModel):
    def name(self):
        return "UASTINet Image Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=5, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')

            parser.add_argument('--lambda_edge', type=float, default=5.0,help='weight for Sobel edge consistency loss')

            parser.add_argument('--lambda_hole', type=float, default=2.0,help='weight for L1 loss inside missing region')

            parser.add_argument('--lambda_context', type=float, default=0.5,help='weight for L1 loss outside missing region')

            parser.add_argument('--texture_num_layers', type=int, default=2,help='Number of stacked TF-Conv layers in texture graph module')
            parser.add_argument('--texture_return_intermediate', action='store_true',help='Return intermediate features from TF-Conv layers')
            parser.add_argument('--lambda_struct',type=float,default=1.0,help='weight for SUPNet structural NLL loss')
        return parser

    def __init__(self, opt):
        """Initial the  model"""
        BaseModel.__init__(self, opt)

        self.visual_names = ['img_m', 'img_truth', 'img_enh', 'merged_image']
        self.value_names = ['u_m', 'sigma_m', 'u_prior', 'sigma_prior']
        self.model_names = ['CSA', 'ET', 'ES', 'SGTFM', 'G', 'D', 'fuse_s', 'fuse_t']
        self.loss_names = ['kl_s', 'struct', 'app_G1', 'edge', 'img_dg', 'ad_l', 'G']
        self.distribution = []

        self.current_iteration = 0
        self.total_iterations = getattr(opt, 'niter', 100000)            

        self.net_ET = network.define_et(init_type='orthogonal', gpu_ids=opt.gpu_ids)        
        self.net_ES = network.define_es(init_type='orthogonal', gpu_ids=opt.gpu_ids)        
        self.net_G = network.define_de(init_type='orthogonal', gpu_ids=opt.gpu_ids)    
        self.net_D = network.define_dis_g(init_type='orthogonal', gpu_ids=opt.gpu_ids)      
        self.net_fuse_s = network.define_fuse_s(init_type='orthogonal', gpu_ids=opt.gpu_ids)          
        self.net_fuse_t = network.define_fuse_t(init_type='orthogonal', gpu_ids=opt.gpu_ids)          
        self.net_CSA = network.define_csa(init_type='orthogonal', gpu_ids=opt.gpu_ids)
        self.net_SGTFM = network.define_sgtfm(init_type='orthogonal', gpu_ids=opt.gpu_ids)
                                        
        self.lossNet = network.VGG16FeatureExtractor()
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            self.lossNet.cuda(opt.gpu_ids[0])
        self.lossNet.eval()
        for p in self.lossNet.parameters():
            p.requires_grad = False

        print(f"net_ET type: {type(self.net_ET)}")

        if hasattr(self.net_ET, "module"):
            print(f"net_ET module type: {type(self.net_ET.module)}")
        else:
            print("net_ET has no .module attribute")

        print(f"net_ET forward method: {self.net_ET.forward}")

        if self.isTrain:                                                
                                                 
            self.GANloss = external_function.GANLoss(opt.gan_mode)                                                         
            self.L1loss = torch.nn.L1Loss()                                            
            self.L2loss = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(
                    filter(lambda p: p.requires_grad, self.net_CSA.parameters()),
                    filter(lambda p: p.requires_grad, self.net_ES.parameters()),
                    filter(lambda p: p.requires_grad, self.net_ET.parameters()),
                    filter(lambda p: p.requires_grad, self.net_SGTFM.parameters()),
                    filter(lambda p: p.requires_grad, self.net_fuse_s.parameters()),
                    filter(lambda p: p.requires_grad, self.net_fuse_t.parameters()),
                    filter(lambda p: p.requires_grad, self.net_G.parameters()),
                ),
                lr=opt.lr,
                betas=(0.0, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),lr=opt.lr, betas=(0.0, 0.999))
                                                                                                   
            self.optimizers.append(self.optimizer_G)                
            self.optimizers.append(self.optimizer_D)
                                                               
        self.setup(opt)

            
    def set_input(self, input):            

        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']                                      
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:                                           
                              
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        self.img_truth = self.img * 2 - 1                                              
        if self.mask.size(1) != 1:
            mask1 = self.mask[:, :1, :, :]
        else:
            mask1 = self.mask

        mask3 = mask1.repeat(1, 3, 1, 1)
        self.img_m = (1 - mask3) * self.img_truth + mask3
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)                                                           
        self.scale_mask = task.scale_pyramid(mask1, self.opt.output_scale)

    def test(self):
        """
        Test / inference for the modified UASTINet pipeline.
        """

        was_training = self.isTrain

        for name in self.model_names:
            if isinstance(name, str) and hasattr(self, "net_" + name):
                net = getattr(self, "net_" + name)
                net.eval()

        with torch.no_grad():
            if self.mask.size(1) != 1:
                mask1 = self.mask[:, :1, :, :]
            else:
                mask1 = self.mask

            if mask1.shape[2:] != self.img_truth.shape[2:]:
                mask1 = F.interpolate(
                    mask1,
                    size=self.img_truth.shape[2:],
                    mode="nearest"
                )
            mask3 = mask1.repeat(1, self.img_truth.size(1), 1, 1)

            self.img_m = (1.0 - mask3) * self.img_truth + mask3

            self.forward()
            if isinstance(self.img_out, list):
                self.img_out = self.img_out[-1]

            if self.img_out.shape[2:] != self.img_truth.shape[2:]:
                self.img_out = F.interpolate(
                    self.img_out,
                    size=self.img_truth.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
            self.merged_image = self.img_truth * (1.0 - mask3) + self.img_out * mask3

            self.save_results(self.img_truth, data_name="truth")
            self.save_results(self.img_m, data_name="mask")
            self.save_results(self.img_out, data_name="raw_out")
            self.save_results(self.merged_image, data_name="out")
            self.save_results(self.merged_image, data_name="merged")

            if hasattr(self, "img_enh") and self.img_enh is not None:
                img_enh = self.img_enh

                if img_enh.shape[2:] != self.img_truth.shape[2:]:
                    img_enh = F.interpolate(
                        img_enh,
                        size=self.img_truth.shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )

                self.save_results(img_enh, data_name="enh")

            if hasattr(self, "S_rec") and self.S_rec is not None:
                s_rec = self.S_rec

                if isinstance(s_rec, list):
                    s_rec = s_rec[-1]

                if s_rec.shape[2:] != self.img_truth.shape[2:]:
                    s_rec = F.interpolate(
                        s_rec,
                        size=self.img_truth.shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )

                if s_rec.size(1) == 1:
                    s_rec = s_rec.repeat(1, 3, 1, 1)

                                                     
                s_min = s_rec.amin(dim=(2, 3), keepdim=True)
                s_max = s_rec.amax(dim=(2, 3), keepdim=True)
                s_rec_vis = (s_rec - s_min) / (s_max - s_min + 1e-6)
                s_rec_vis = s_rec_vis * 2.0 - 1.0

                self.save_results(s_rec_vis, data_name="structure")

            if hasattr(self, "delta") and self.delta is not None:
                delta = self.delta

                if isinstance(delta, list):
                    delta = delta[-1]

                if delta.shape[2:] != self.img_truth.shape[2:]:
                    delta = F.interpolate(
                        delta,
                        size=self.img_truth.shape[2:],
                        mode="bilinear",
                        align_corners=False
                    )

                                                     
                if delta.size(1) == 1:
                    delta_vis = delta.repeat(1, 3, 1, 1)
                else:
                    delta_vis = delta[:, :3, :, :]

                                                      
                d_min = delta_vis.amin(dim=(2, 3), keepdim=True)
                d_max = delta_vis.amax(dim=(2, 3), keepdim=True)
                delta_vis = (delta_vis - d_min) / (d_max - d_min + 1e-6)
                delta_vis = delta_vis * 2.0 - 1.0

                self.save_results(delta_vis, data_name="uncertainty")

        if was_training:
            for name in self.model_names:
                if isinstance(name, str) and hasattr(self, "net_" + name):
                    net = getattr(self, "net_" + name)
                    net.train()
        return

    def get_distribution(self, distributions):
        q_distribution, kl = 0, 0
        self.distribution = []

        for i, distribution in enumerate(distributions):

            if isinstance(distribution, list) and len(distribution) == 1:
                distribution = distribution[0]

                             
            q_mu, q_sigma = distribution

                                  
            q_sigma = F.softplus(q_sigma) + 1e-6                        

                                
            m_distribution = torch.distributions.Normal(torch.zeros_like(q_mu), torch.ones_like(q_sigma))
                                   
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

                      
            kl += torch.distributions.kl_divergence(q_distribution, m_distribution)

                    
            self.distribution.append([torch.zeros_like(q_mu), torch.ones_like(q_sigma), q_mu, q_sigma])

        return kl

    def forward(self):

        mask = self.mask

        if mask.size(1) != 1:
            mask1 = mask[:, :1, :, :]
        else:
            mask1 = mask

        if mask1.shape[2:] != self.img_truth.shape[2:]:
            mask1 = F.interpolate(
                mask1,
                size=self.img_truth.shape[2:],
                mode='nearest'
            )

        mask1 = (mask1 > 0.5).float()
        mask3 = mask1.repeat(1, 3, 1, 1)

        self.img_m = (1.0 - mask3) * self.img_truth + mask3

        self.img_enh = self.net_CSA(self.img_m)

        if hasattr(self.net_ES, "module"):
            S_gt = self.net_ES.module.build_structure_gt(self.img_truth)
        else:
            S_gt = self.net_ES.build_structure_gt(self.img_truth)

        sup_input = torch.cat([self.img_enh, mask3], dim=1)

        sup_out = self.net_ES(
            sup_input,
            mask1,
            self.current_iteration,
            self.total_iterations,
            S_gt=S_gt
        )

        self.s_feature = sup_out["features"]
        self.delta = sup_out["delta"]
        self.S_rec = sup_out["S_rec"]
        self.loss_struct_raw = sup_out["loss_struct"]

        et_input = torch.cat([self.img_enh, mask3], dim=1)
        t_x, self.t_feature = self.net_ET(et_input, self.delta)

        s_feature_for_fuse = [self.s_feature[-1]]
        fuse_s = self.net_fuse_s(s_feature_for_fuse)
        fuse_t = self.net_fuse_t(self.t_feature)

        fused = self.net_SGTFM(self.s_feature, self.t_feature, self.delta)

        mu, sigma = torch.split(fused, 256, dim=1)
        sigma = F.softplus(sigma) + 1e-6

        self.kl_g_s = self.get_distribution([[mu, sigma]])

        distribution_normal = torch.distributions.Normal(mu, sigma)
        z = distribution_normal.rsample()

        self.img_g = self.net_G(z, fuse_s, fuse_t)

        if isinstance(self.img_g, list):
            self.img_out = self.img_g[-1]
        else:
            self.img_out = self.img_g

        if self.img_out.shape[2:] != self.img_truth.shape[2:]:
            self.img_out = F.interpolate(
                self.img_out,
                size=self.img_truth.shape[2:],
                mode='bilinear',
                align_corners=False
            )
                                                                      
        self.merged_image = self.img_truth * (1.0 - mask3) + self.img_out * mask3


    def backward_D_basic(self, netD, real, fake):                 
        """Calculate GAN loss for the discriminator"""
                
        D_real = netD(real)                                     
        D_real_loss = self.GANloss(D_real, True, True)                             
                      
        D_fake = netD(fake.detach())                                                    
        D_fake_loss = self.GANloss(D_fake, False, True)                           
                                          
        D_loss = (D_real_loss + D_fake_loss) * 0.5                                        

        D_loss.backward()                                                  

        return D_loss          

    def _to_gray_01(self, x):
        """
        Convert RGB image from [-1, 1] or [0, 1] to grayscale [0, 1].
        x: [B, 3, H, W]
        """
        x = x[:, :3, :, :]

        if x.min() < 0:
            x = (x + 1.0) / 2.0

        x = torch.clamp(x, 0.0, 1.0)

        gray = (
                0.299 * x[:, 0:1, :, :] +
                0.587 * x[:, 1:2, :, :] +
                0.114 * x[:, 2:3, :, :]
        )

        return gray

    def _sobel_edge(self, x):
        """
        Sobel edge map for mural line / structure consistency.
        Return: [B, 1, H, W]
        """
        gray = self._to_gray_01(x)

        sobel_x = gray.new_tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).view(1, 1, 3, 3)

        sobel_y = gray.new_tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).view(1, 1, 3, 3)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return edge

    def _masked_l1(self, pred, target, weight):
        """
        Normalized masked L1.
        This avoids loss scale changing too much with mask area.
        """
        return (torch.abs(pred - target) * weight).sum() / (weight.sum() + 1e-6)

    def backward_D(self):
        """Calculate the GAN loss for the discriminator"""
        base_function._unfreeze(self.net_D)
        self.loss_img_dg = self.backward_D_basic(
            self.net_D,
            self.img_truth,
            self.merged_image
        )

    def backward_G(self):
        """
        Calculate training loss for the generator.
        """
        base_function._freeze(self.net_D)

        device = self.img_truth.device

        if self.mask.size(1) != 1:
            mask1 = self.mask[:, :1, :, :]
        else:
            mask1 = self.mask

        if mask1.shape[2:] != self.img_truth.shape[2:]:
            mask1 = F.interpolate(
                mask1,
                size=self.img_truth.shape[2:],
                mode='nearest'
            )

        mask1 = (mask1 > 0.5).float()
        mask3 = mask1.repeat(1, self.img_truth.size(1), 1, 1)

        if isinstance(self.img_out, list):
            fake_img = self.img_out[-1]
        else:
            fake_img = self.img_out

        if fake_img.shape[2:] != self.img_truth.shape[2:]:
            fake_img = F.interpolate(
                fake_img,
                size=self.img_truth.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        self.img_out = fake_img
        self.merged_image = self.img_truth * (1.0 - mask3) + self.img_out * mask3
        if hasattr(self, "kl_g_s") and self.kl_g_s is not None:
            self.loss_kl_s = self.kl_g_s.mean() * self.opt.lambda_kl
        else:
            self.loss_kl_s = torch.tensor(0.0, device=device)

        if hasattr(self, "loss_struct_raw") and self.loss_struct_raw is not None:
            lambda_struct = getattr(self.opt, "lambda_struct", 1.0)
            self.loss_struct = self.loss_struct_raw * lambda_struct
        else:
            self.loss_struct = torch.tensor(0.0, device=device)

        D_fake_g = self.net_D(self.merged_image)
        D_real_g = self.net_D(self.img_truth.detach())

        self.loss_ad_l = self.L2loss(
            D_fake_g,
            D_real_g.detach()
        ) * self.opt.lambda_g

        loss_app_hole = self._masked_l1(
            self.img_out,
            self.img_truth,
            mask3
        )
        loss_app_context = self._masked_l1(
            self.img_out,
            self.img_truth,
            1.0 - mask3
        )

        lambda_hole = getattr(self.opt, "lambda_hole", 2.0)
        lambda_context = getattr(self.opt, "lambda_context", 0.5)

        self.loss_app_G1 = (lambda_hole * loss_app_hole +lambda_context * loss_app_context) * self.opt.lambda_rec

        edge_fake = self._sobel_edge(self.merged_image)
        edge_real = self._sobel_edge(self.img_truth)

        edge_mask = F.max_pool2d(
            mask1,
            kernel_size=7,
            stride=1,
            padding=3
        )

        self.loss_edge = self._masked_l1(
            edge_fake,
            edge_real.detach(),
            edge_mask
        ) * getattr(self.opt, "lambda_edge", 5.0)

        real_feats = self.lossNet(self.img_truth)
        comp_feats = self.lossNet(self.merged_image)

        self.loss_G_style = base_function.style_loss(
            real_feats,
            comp_feats
        )

        self.loss_G_content = base_function.perceptual_loss(
            real_feats,
            comp_feats
        )

        self.loss_G = 0.05 * self.loss_G_content + 120.0 * self.loss_G_style

        self.loss_app_G2 = torch.tensor(0.0, device=device)

        total_loss = (
                self.loss_kl_s +
                self.loss_struct +
                self.loss_app_G1 +
                self.loss_edge +
                self.loss_ad_l +
                self.loss_G
        )

        self.loss_G_total = total_loss

        total_loss.backward()


    def optimize_parameters(self):     

        """update network weights"""
                                                        
        self.forward()        
                                                                  
        self.optimizer_D.zero_grad()                                        
        self.backward_D()                           
        self.optimizer_D.step()                                                                  
                                                              
        self.optimizer_G.zero_grad()                                
        self.backward_G()                           
        self.optimizer_G.step()                                                                  
