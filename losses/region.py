import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionLoss(nn.Module):
    """Region Consistency Loss for the G_E (crack elimination) branch only. [paper Eq.11-12]
    
    Computes an attention mask M_attn from the difference in D_N intermediate features
    between the real normal image and the generated crack-free image.
    
    M_attn = sigmoid(||F(I_N) - F(G_E(I_C))||_channel_mean)
    
    L_region = ||Sobel(G_E(I_C)) - Sobel(I_C)||_1         (gradient preservation)
             + ||(G_E(I_C) - I_C) * (1 - M_attn)||_1     (pixel diff in non-crack region)
    
    NOTE: L1 loss throughout — no L2/MSE as in the paper formulation (Eq.12).
    Applied ONLY to the G_E direction. NOT applied to G_A.
    """

    def __init__(self):
        super(RegionLoss, self).__init__()
        # Sobel kernels for differentiable gradient computation
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('gx', gx)
        self.register_buffer('gy', gy)

    def sobel_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude (differentiable). [paper Eq.12]"""
        b, c, h, w = x.size()
        x_r = x.view(b * c, 1, h, w)
        grad_x = F.conv2d(x_r, self.gx, padding=1)
        grad_y = F.conv2d(x_r, self.gy, padding=1)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return magnitude.view(b, c, h, w)

    def forward(self, fake: torch.Tensor, real: torch.Tensor,
                feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fake: G_E(I_C) — generated crack-free image, shape (B, 3, H, W).
            real: I_C — original cracked image, shape (B, 3, H, W).
            feat_real: D_N intermediate features from real normal image F(I_N), shape (B, 512, H', W').
            feat_fake: D_N intermediate features from generated image F(G_E(I_C)), shape (B, 512, H', W').
        
        Returns:
            Scalar region loss value.
        """
        # --- Attention Mask [paper Eq.11] ---
        # M_attn = sigmoid(mean channel-wise L1 diff of discriminator features)
        feat_diff = torch.abs(feat_real - feat_fake)
        # Collapse channel dim to single spatial map
        feat_diff_mean = feat_diff.mean(dim=1, keepdim=True)  # (B, 1, H', W')
        # Upsample to match image size
        m_attn = torch.sigmoid(
            F.interpolate(feat_diff_mean, size=fake.shape[2:], mode='bilinear', align_corners=False)
        )  # (B, 1, H, W)

        # --- Gradient preservation term [paper Eq.12, first term] ---
        sobel_fake = self.sobel_gradient(fake)
        sobel_real = self.sobel_gradient(real)
        loss_grad = F.l1_loss(sobel_fake, sobel_real)

        # --- Pixel preservation in non-crack regions [paper Eq.12, second term] ---
        # (1 - M_attn) ≈ 1 in non-crack regions → enforce pixel preservation (L1)
        diff = fake - real
        loss_pixel = torch.mean(torch.abs((1 - m_attn) * diff))

        return loss_grad + loss_pixel
