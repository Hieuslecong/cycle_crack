"""loss.py — All 5 Cycle-Crack Loss Functions (v3, paper-accurate)

Losses:
  1. IdentityLoss   [paper Eq.3]
  2. LSGANLoss      [paper Eq.4]
  3. CycleLoss      [paper Eq.5]
  4. TextureLoss    [paper §III-C]: MS-SSIM + MSGMS + Style (VGG-16 layer l=3)
  5. RegionLoss     [paper Eq.11-12]: attention mask + Sobel + pixel diff
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── 1. Identity Loss ────────────────────────────────────────────────── Eq.3 ──

class IdentityLoss(nn.Module):
    """L_idt = λ_GE‖G_E(I_N)-I_N‖₁ + λ_GA‖G_A(I_C)-I_C‖₁"""
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


# ── 2. Adversarial Loss (LSGAN) ─────────────────────────────────────── Eq.4 ──

class LSGANLoss(nn.Module):
    """Least-Squares GAN loss using MSE."""
    def forward(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


# ── 3. Cycle Consistency Loss ───────────────────────────────────────── Eq.5 ──

class CycleLoss(nn.Module):
    """L_cycle = λ_C‖I''_C - I_C‖₁ + λ_N‖I''_N - I_N‖₁"""
    def forward(self, reconstructed, real):
        return F.l1_loss(reconstructed, real)


# ── 4. Texture Consistency Loss ─────────────────────────────────── §III-C ──

def _gaussian_window(size: int, sigma: float, channels: int) -> torch.Tensor:
    g = torch.tensor([math.exp(-(x - size//2)**2 / (2*sigma**2)) for x in range(size)])
    g = g / g.sum()
    w = g.unsqueeze(1).mm(g.unsqueeze(0)).float()
    return w.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def _ssim(x, y, window, size, channels):
    mu1 = F.conv2d(x, window, padding=size//2, groups=channels)
    mu2 = F.conv2d(y, window, padding=size//2, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1 = F.conv2d(x*x, window, padding=size//2, groups=channels) - mu1_sq
    s2 = F.conv2d(y*y, window, padding=size//2, groups=channels) - mu2_sq
    s12= F.conv2d(x*y, window, padding=size//2, groups=channels) - mu12
    C1, C2 = 0.01**2, 0.03**2
    return ((2*mu12+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss, S=3 scales. [paper Eq.6-7]"""
    def __init__(self, window_size=11, scales=3):
        super().__init__()
        self.ws, self.scales = window_size, scales

    def forward(self, x, y):
        # Normalize from [-1,1] to [0,1]
        x, y = (x+1)/2, (y+1)/2
        c = x.shape[1]
        w = _gaussian_window(self.ws, 1.5, c).to(x.device).type_as(x)
        loss = 0.0
        for _ in range(self.scales):
            loss += 1 - _ssim(x, y, w, self.ws, c).mean()
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        return loss / self.scales


class MSGMSLoss(nn.Module):
    """Multi-Scale Gradient Magnitude Similarity loss, S=3. [paper Eq.8]
    Sobel gradient via F.conv2d (differentiable).
    """
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def _gm(self, x):
        b,c,h,w = x.shape
        x = x.view(b*c,1,h,w)
        gmx = F.conv2d(x, self.gx, padding=1)
        gmy = F.conv2d(x, self.gy, padding=1)
        return torch.sqrt(gmx**2 + gmy**2 + 1e-6).view(b,c,h,w)

    def forward(self, x, y):
        x, y = (x+1)/2, (y+1)/2
        loss = 0.0
        for _ in range(3):
            loss += F.l1_loss(self._gm(x), self._gm(y))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        return loss / 3.0


class StyleLoss(nn.Module):
    """VGG-16 Style loss — layer l=3 only. [paper §III-C, Eq.9-10]

    Paper: "with l=3 used in this work" → single VGG-16 feature map at index 3.
    VGG-16 features[:4] = [Conv2d, ReLU, Conv2d, ReLU] → output at relu1_2.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg_layer3 = nn.Sequential(*list(vgg.children())[:4])  # single layer l=3
        for p in self.vgg_layer3.parameters():
            p.requires_grad = False  # freeze

    @staticmethod
    def gram(x):
        b, c, h, w = x.shape
        f = x.view(b, c, h*w)
        return f.bmm(f.transpose(1, 2)) / (c * h * w)

    def forward(self, x, y):
        # Normalize from [-1,1] to ImageNet range
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        xn = ((x+1)/2 - mean) / std
        yn = ((y+1)/2 - mean) / std
        # Texture loss is computed in FP32 so no manual downscaling is needed to prevent overflow
        return F.mse_loss(self.gram(self.vgg_layer3(xn)),
                          self.gram(self.vgg_layer3(yn)))


class TextureLoss(nn.Module):
    """L_text = L_ms_ssim + L_msgms + L_style  [paper §III-C]

    Applied to BOTH cycle directions:
      (I_C, I''_C) and (I_N, I''_N)
    """
    def __init__(self):
        super().__init__()
        self.msssim = MSSSIMLoss(scales=3)
        self.msgms  = MSGMSLoss()
        self.style  = StyleLoss()

    def forward(self, rec, real):
        return self.msssim(rec, real) + self.msgms(rec, real) + self.style(rec, real)


# ── 5. Region Consistency Loss ──────────────────────────────── Eq.11-12 ──

class RegionLoss(nn.Module):
    """L_region — applied to G_E branch ONLY. [paper Eq.11-12]

    Attention mask from D_N feature difference highlights crack regions.
    Loss = Sobel gradient preservation + pixel diff in non-crack regions.
    """
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def _sobel(self, x):
        """Differentiable Sobel magnitude via F.conv2d. [paper Eq.12]"""
        b,c,h,w = x.shape
        xr = x.view(b*c,1,h,w)
        gx = F.conv2d(xr, self.gx, padding=1)
        gy = F.conv2d(xr, self.gy, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6).view(b,c,h,w)

    def forward(self, fake, real, feat_real, feat_fake):
        """
        Args:
            fake     : G_E(I_C), generated crack-free image.
            real     : I_C, original cracked image.
            feat_real: D_N intermediate features from REAL normal image D_N(I_N).
            feat_fake: D_N intermediate features from generated image D_N(G_E(I_C)).
        """
        # Attention mask [paper Eq.11]:
        # M_attn = σ( ||F(I_N) - F(G_E(I_C))|| ) — diff from D_N features
        m = torch.sigmoid(
            F.interpolate(
                torch.abs(feat_real - feat_fake).mean(dim=1, keepdim=True),
                size=fake.shape[2:], mode='bilinear', align_corners=False
            )
        )  # (B,1,H,W)

        # Gradient preservation [paper Eq.12, term 1]
        loss_grad = F.l1_loss(self._sobel(fake), self._sobel(real))

        # Pixel preservation in non-crack regions [paper Eq.12, term 2]
        loss_pix = F.l1_loss((fake - real) * (1.0 - m), torch.zeros_like(fake))

        return loss_grad + loss_pix
