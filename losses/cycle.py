import torch.nn as nn

class CycleLoss(nn.Module):
    """Defines the cycle consistency loss function (L1)"""
    
    def __init__(self):
        super(CycleLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, reconstructed, real):
        """Calculate loss given reconstructed and real images"""
        return self.loss(reconstructed, real)
