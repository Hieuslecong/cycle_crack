import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator as described in the paper.
    
    Exactly 5 convolutional layers. No spectral normalization (not in paper).
    BatchNorm2d on layers 2, 3, 4 (not on layer 1 or 5).
    forward() returns both the prediction and intermediate features from layer 4
    for use in the Region Consistency Loss.
    """
    
    def __init__(self, in_channels: int = 3, ndf: int = 64):
        """Construct a 70x70 PatchGAN discriminator.
        
        Args:
            in_channels (int): Number of channels in input images.
            ndf (int): Number of filters in the first conv layer.
        """
        super(PatchGANDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        
        # Layer 1: Conv → LeakyReLU (no BN)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        )
        
        # Layer 2: Conv → BN → LeakyReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        
        # Layer 3: Conv → BN → LeakyReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        
        # Layer 4: Conv → BN → LeakyReLU  [used for attention mask / Region Loss]
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        
        # Layer 5: Conv → (no BN, no activation)
        self.layer5 = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x (Tensor): Input image tensor (B, C, H, W).
            
        Returns:
            out (Tensor): Patch-level real/fake predictions.
            feat (Tensor): Intermediate feature map from layer 4, shape (B, 512, H', W').
                           Used to compute the attention mask M_attn in Region Loss.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)   # shape: (B, ndf*8, H', W')
        out = self.layer5(feat)
        return out, feat
