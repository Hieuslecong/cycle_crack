import torch
import torch.nn as nn

class LSGANLoss(nn.Module):
    """Defines the LSGAN adversarial loss function"""
    
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the prediction"""
        if target_is_real:
            target_tensor = torch.tensor(1.0, requires_grad=False)
        else:
            target_tensor = torch.tensor(0.0, requires_grad=False)
        return target_tensor.expand_as(prediction).to(prediction.device)

    def forward(self, prediction, target_is_real):
        """Calculate loss given prediction and target"""
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
