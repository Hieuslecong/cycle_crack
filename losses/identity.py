import torch.nn as nn

class IdentityLoss(nn.Module):
    """Defines the identity loss function (L1)"""
    
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, idt_generated, real):
        """Calculate loss given identity-generated and real images"""
        return self.loss(idt_generated, real)
