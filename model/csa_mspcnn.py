import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAMSPCNN(nn.Module):
    """
    CSA-MSPCNN enhancement module.

    Input:
        x: [B, 3, H, W], range [-1, 1]

    Output:
        enhanced: [B, 3, H, W], range [-1, 1]

    Note:
        The CSA firing response is computed on the luminance channel and then
        applied to the RGB image as a residual enhancement map. This avoids
        pseudo-color artifacts caused by directly visualizing firing responses
        as RGB images.
    """

    def __init__(self, iterations=8, eta=0.5, sharpness=0.30, otsu_bins=256, eps=1e-6):
        super().__init__()
        self.iterations = iterations
        self.eta = eta
        self.sharpness = sharpness
        self.otsu_bins = otsu_bins
        self.eps = eps

        M_base = torch.tensor(
            [[0.05, 0.20, 0.05],
             [0.20, 0.00, 0.20],
             [0.05, 0.20, 0.05]],
            dtype=torch.float32
        )

        W_base = torch.tensor(
            [[0.10, 0.15, 0.10],
             [0.15, 0.00, 0.15],
             [0.10, 0.15, 0.10]],
            dtype=torch.float32
        )

        self.register_buffer("M_base", M_base.view(1, 1, 3, 3))
        self.register_buffer("W_base", W_base.view(1, 1, 3, 3))

    @torch.no_grad()
    def _otsu_threshold(self, gray):
        """
        Compute global Otsu threshold S' for each image.

        Args:
            gray: [B, 1, H, W], range [0, 1]

        Returns:
            threshold: [B, 1, 1, 1]
        """
        b = gray.shape[0]
        thresholds = []

        for i in range(b):
            values = gray[i].reshape(-1).detach().clamp(0, 1)

            hist = torch.histc(values, bins=self.otsu_bins, min=0.0, max=1.0)
            prob = hist / hist.sum().clamp_min(self.eps)

            bin_centers = torch.linspace(
                0.0,
                1.0,
                self.otsu_bins,
                device=gray.device,
                dtype=gray.dtype,
            )

            omega = torch.cumsum(prob, dim=0)
            mu = torch.cumsum(prob * bin_centers, dim=0)
            mu_total = mu[-1]

            sigma_b = (mu_total * omega - mu) ** 2
            sigma_b = sigma_b / (omega * (1.0 - omega)).clamp_min(self.eps)

            idx = torch.argmax(sigma_b)
            threshold = bin_centers[idx].clamp(self.eps, 1.0)
            thresholds.append(threshold)

        return torch.stack(thresholds).view(b, 1, 1, 1)

    def _normalize_response(self, r):
        """Normalize a response map to [0, 1] per image."""
        r_min = r.amin(dim=(1, 2, 3), keepdim=True)
        r_max = r.amax(dim=(1, 2, 3), keepdim=True)
        return (r - r_min) / (r_max - r_min).clamp_min(self.eps)

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], but got {x.shape}")

        # [-1, 1] -> [0, 1]
        s_rgb = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

        # Compute the CSA firing dynamics on the luminance channel to avoid
        # channel-wise firing differences that may introduce color shifts.
        r, g, b = s_rgb[:, 0:1], s_rgb[:, 1:2], s_rgb[:, 2:3]
        s = 0.299 * r + 0.587 * g + 0.114 * b  # [B, 1, H, W]

        # Global Otsu threshold S'
        s_prime = self._otsu_threshold(s).to(dtype=s.dtype, device=s.device)

        # Eq. (7)
        alpha = torch.log(1.0 / s_prime.clamp_min(self.eps))

        exp_1a = torch.exp(-alpha)
        exp_2a = torch.exp(-2.0 * alpha)
        exp_3a = torch.exp(-3.0 * alpha)
        exp_4a = torch.exp(-4.0 * alpha)
        exp_5a = torch.exp(-5.0 * alpha)

        V = exp_1a + exp_2a + exp_3a
        Q = exp_1a
        C_param = exp_3a + exp_4a

        # Initial states
        y = torch.zeros_like(s)
        u_prev = torch.zeros_like(s)
        e = s_prime.expand_as(s).clone()

        M = self.M_base.to(dtype=s.dtype, device=s.device)
        W = self.W_base.to(dtype=s.dtype, device=s.device)

        firing_accum = torch.zeros_like(s)

        for _ in range(self.iterations):
            # Eq. (8): structure-oriented and texture-oriented neighborhood responses
            y_nei_m = exp_4a * F.conv2d(y, M, padding=1)
            y_nei_w = exp_5a * F.conv2d(y, W, padding=1)

            # Eqs. (1)--(3)
            u1 = exp_3a * u_prev + s * (1.0 + y_nei_m)
            u2 = exp_4a * u_prev + s * y_nei_w
            u3 = exp_5a * u_prev
            u = u1 + u2 + u3

            # Eq. (4)
            y = (u > e).to(dtype=s.dtype)

            # Eqs. (5)--(6)
            e1 = exp_1a * e + V * y
            e2 = exp_2a * e + C_param * Q
            e = e1 + e2

            u_prev = u
            firing_accum = firing_accum + y

        # CSA firing response from accumulated firing maps
        response = self._normalize_response(firing_accum)
        response_centered = response - response.mean(dim=(2, 3), keepdim=True)

        # Lightweight high-frequency residual to preserve fine mural details
        blur_rgb = F.avg_pool2d(s_rgb, kernel_size=3, stride=1, padding=1)
        detail_rgb = s_rgb - blur_rgb

        # Residual enhancement: CSA response + high-frequency detail
        enhanced = (
            s_rgb
            + self.eta * response_centered.expand_as(s_rgb)
            + self.sharpness * detail_rgb
        )
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced * 2.0 - 1.0


def compute_sobel_gradient(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute raw Sobel gradient magnitude.

    Args:
        img: [B, 3, H, W], range [-1, 1] or [0, 1]

    Returns:
        grad: [B, 1, H, W], raw gradient magnitude
    """
    if img.min() < 0:
        img = (img + 1.0) / 2.0

    img = torch.clamp(img, 0.0, 1.0)

    gray = (
        0.299 * img[:, 0:1, :, :]
        + 0.587 * img[:, 1:2, :, :]
        + 0.114 * img[:, 2:3, :, :]
    )

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=img.dtype,
        device=img.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=img.dtype,
        device=img.device,
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)
    return grad


def normalize_pair(grad_before: torch.Tensor, grad_after: torch.Tensor, eps: float = 1e-6):
    """Jointly normalize before/after gradient maps to make them comparable."""
    g_min = torch.minimum(
        grad_before.amin(dim=(1, 2, 3), keepdim=True),
        grad_after.amin(dim=(1, 2, 3), keepdim=True),
    )
    g_max = torch.maximum(
        grad_before.amax(dim=(1, 2, 3), keepdim=True),
        grad_after.amax(dim=(1, 2, 3), keepdim=True),
    )

    grad_before = (grad_before - g_min) / (g_max - g_min).clamp_min(eps)
    grad_after = (grad_after - g_min) / (g_max - g_min).clamp_min(eps)

    return grad_before, grad_after