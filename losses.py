import torch

import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, recon_loss_type='mse', reduction='sum'):
        super(VAELoss, self).__init__()
        self.recon_loss_type = recon_loss_type
        self.reduction = reduction

    def forward(self, recon_x, x, mu, logvar):
        if self.recon_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
        elif self.recon_loss_type == 'mse':
            # check reduction
            recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)
        else :
            recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)

            
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld