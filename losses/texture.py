import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor,
          window_size: int, channel: int, size_average: bool = True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss — 3 scales, 11×11 window. [paper Eq.6-7]"""
    
    def __init__(self, window_size: int = 11, num_scales: int = 3):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.num_scales = num_scales
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # Normalize to [0, 1]
        img1_norm = (img1 + 1) / 2
        img2_norm = (img2 + 1) / 2

        loss = 0.0
        for _ in range(self.num_scales):
            sim = _ssim(img1_norm, img2_norm, window, self.window_size, channel)
            loss += (1 - sim)
            img1_norm = F.avg_pool2d(img1_norm, (2, 2))
            img2_norm = F.avg_pool2d(img2_norm, (2, 2))

        return loss / self.num_scales


class MSGMSLoss(nn.Module):
    """Multi-Scale Gradient Magnitude Similarity Loss — 3 scales. [paper Eq.8]"""
    
    def __init__(self):
        super(MSGMSLoss, self).__init__()
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def get_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x_reshaped = x.view(b * c, 1, h, w)
        grad_x = F.conv2d(x_reshaped, self.gx, padding=1)
        grad_y = F.conv2d(x_reshaped, self.gy, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return magnitude.view(b, c, h, w)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        y = (y + 1) / 2
        loss = 0.0
        for _ in range(3):
            mag_x = self.get_gradient_magnitude(x)
            mag_y = self.get_gradient_magnitude(y)
            loss += F.l1_loss(mag_x, mag_y)
            x = F.avg_pool2d(x, (2, 2))
            y = F.avg_pool2d(y, (2, 2))
        return loss / 3.0


class StyleLoss(nn.Module):
    """VGG-16 Style (Gram matrix) Loss — single layer l=3 only. [paper Eq.9-10]
    
    CRITICAL: Paper states l=3 (single VGG layer). Uses VGG-16, NOT VGG-19.
    Do NOT use multiple VGG layers.
    """
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        # VGG-16, layer index 3 ('relu1_2' equivalent) [paper §III-C]
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # Slice up to and including layer index 3 (0-indexed)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:4])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def gram_matrix(self, y: torch.Tensor) -> torch.Tensor:
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet mean/std from [-1,1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake.device)
        fake_norm = ((fake + 1) / 2 - mean) / std
        real_norm = ((real + 1) / 2 - mean) / std

        feat_fake = self.feature_extractor(fake_norm)
        feat_real = self.feature_extractor(real_norm)

        # [not in paper]: Scale by 1e-3 — Gram MSE overflows fp16 AMP without it.
        loss = F.mse_loss(self.gram_matrix(feat_fake), self.gram_matrix(feat_real)) * 1e-3
        return loss


class TextureLoss(nn.Module):
    """Texture Consistency Loss: MS-SSIM + MSGMS + VGG-16 Style. [paper §III-C]
    
    Applied to BOTH cycle branches:
      - Cycle C→N→C: pair (I_C, I_C_rec)
      - Cycle N→C→N: pair (I_N, I_N_rec)
    
    L_text = L_ms_ssim + L_msgms + L_style  (no separate sub-weights per paper)
    """
    
    def __init__(self):
        super(TextureLoss, self).__init__()
        self.msssim = MSSSIMLoss(num_scales=3)
        self.msgms = MSGMSLoss()
        self.style = StyleLoss()

    def forward(self, reconstructed: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reconstructed: The cycle-reconstructed image.
            real: The original image from that domain.
        """
        l_ssim = self.msssim(reconstructed, real)
        l_gms = self.msgms(reconstructed, real)
        l_style = self.style(reconstructed, real)
        return l_ssim + l_gms + l_style
