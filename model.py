"""model.py — Generator (UNet-256) + Discriminator (PatchGAN-70)

Architecture as described in paper §III-A.
Generator uses BatchNorm2d (NOT InstanceNorm2d).
Discriminator has 5 Conv layers, no spectral normalization.
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Generator: UNet-256                                [paper §III-A]
# ─────────────────────────────────────────────────────────────

class UNet256(nn.Module):
    """UNet-256 generator.

    Uses BatchNorm2d (not InstanceNorm) and skip-connections via concatenation.
    Dropout(0.5) applied to the 3 innermost decoder layers.

    Args:
        in_channels  (int): RGB input channels (default 3).
        out_channels (int): RGB output channels (default 3).
        ngf          (int): Base number of generator filters (default 64).
        num_downs    (int): Number of downsampling layers (default 8 for 256×256).
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=64, num_downs=8):
        super().__init__()
        norm = nn.BatchNorm2d  # [paper §III-A]: BatchNorm2d, NOT InstanceNorm2d

        # Build UNet inside-out
        block = _UNetBlock(ngf*8, ngf*8, submodule=None, norm=norm, innermost=True)
        for _ in range(num_downs - 5):
            block = _UNetBlock(ngf*8, ngf*8, submodule=block, norm=norm, dropout=True)
        block = _UNetBlock(ngf*4, ngf*8, submodule=block, norm=norm)
        block = _UNetBlock(ngf*2, ngf*4, submodule=block, norm=norm)
        block = _UNetBlock(ngf,   ngf*2, submodule=block, norm=norm)
        self.model = _UNetBlock(out_channels, ngf, input_nc=in_channels,
                                submodule=block, norm=norm, outermost=True)

    def forward(self, x):
        return self.model(x)


class _UNetBlock(nn.Module):
    """One UNet encoder-decoder block with optional skip connection."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm=nn.BatchNorm2d, dropout=False):
        super().__init__()
        self.outermost = outermost
        bias = False  # BatchNorm absorbs bias
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, 4, stride=2, padding=1, bias=bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm(inner_nc)
        uprelu   = nn.ReLU(True)
        upnorm   = norm(outer_nc)

        if outermost:
            up   = [uprelu, nn.ConvTranspose2d(inner_nc*2, outer_nc, 4, stride=2, padding=1), nn.Tanh()]
            down = [downconv]
            seq  = down + [submodule] + up
        elif innermost:
            up   = [uprelu, nn.ConvTranspose2d(inner_nc, outer_nc, 4, stride=2, padding=1, bias=bias), upnorm]
            down = [downrelu, downconv]
            seq  = down + up
        else:
            up   = [uprelu, nn.ConvTranspose2d(inner_nc*2, outer_nc, 4, stride=2, padding=1, bias=bias), upnorm]
            down = [downrelu, downconv, downnorm]
            seq  = down + [submodule] + up
            if dropout:
                seq += [nn.Dropout(0.5)]  # 3 innermost decoder layers

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)  # skip connection


def init_weights(net, gain=0.02):
    """Initialize network weights with Normal(0, gain)."""
    def init_fn(m):
        cls = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in cls or 'Linear' in cls):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in cls:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_fn)
    print("initialize network with normal")
    return net


# ─────────────────────────────────────────────────────────────
# Discriminator: PatchGAN-70                         [paper §III-A]
# ─────────────────────────────────────────────────────────────

class PatchGAN70(nn.Module):
    """70×70 PatchGAN discriminator.

    Exactly 5 convolutional layers.
    BatchNorm2d on layers 2, 3, 4 (NOT on layer 1 or 5).
    No spectral normalization — not mentioned in paper.
    forward() returns (patch_pred, feat) where feat is layer-4 output
    used by the Region Consistency Loss.
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        k, p = 4, 1
        # Layer 1: Conv → LeakyReLU  (no BN)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, k, stride=2, padding=p),
            nn.LeakyReLU(0.2, True)
        )
        # Layer 2: Conv → BN → LeakyReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, k, stride=2, padding=p, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True)
        )
        # Layer 3: Conv → BN → LeakyReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, k, stride=2, padding=p, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True)
        )
        # Layer 4: Conv → BN → LeakyReLU  [features used by L_region]
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, k, stride=1, padding=p, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True)
        )
        # Layer 5: Conv  (no BN, no activation)
        self.layer5 = nn.Conv2d(ndf*8, 1, k, stride=1, padding=p)

    def forward(self, x):
        """Returns (patch_prediction, layer4_features)."""
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x)
        feat = self.layer4(x)   # (B, ndf*8, H', W') — used by Region Loss
        out  = self.layer5(feat)
        return out, feat
