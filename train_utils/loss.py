# The original code is licensed under MIT License, which is can be found at licenses/LICENSE_UVIT.txt.

import torch
import torch.nn.functional as F

from utils import *
from train_utils.helper import unwrap_model


# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

def MMDLoss(self, x, y, kernel='rbf', sigma=1.0):
     """Compute Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        x (torch.Tensor): Samples from distribution P.
        y (torch.Tensor): Samples from distribution Q.
        kernel (str): Kernel type ('rbf' or 'linear').
        sigma (float): Kernel bandwidth for RBF kernel.

    Returns:
        torch.Tensor: Computed MMD loss.
        """
    if kernel == 'rbf':
        xx, yy, xy = self._rbf_kernel(x, x, sigma), self._rbf_kernel(y, y, sigma), self._rbf_kernel(x, y, sigma)
    elif kernel == 'linear':
        xx, yy, xy = self._linear_kernel(x, x), self._linear_kernel(y, y), self._linear_kernel(x, y)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()

Losses = {
    'edm': MMDLoss
}


# ----------------------------------------------------------------------------

def patchify(imgs, patch_size=2, num_channels=4):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p, c = patch_size, num_channels
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x


def mae_loss(net, target, pred, mask, norm_pix_loss=True):
    target = patchify(target, net.model.patch_size, net.model.out_channels)
    pred = patchify(pred, net.model.patch_size, net.model.out_channels)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches, (N)
    assert loss.ndim == 1
    return loss

def __init__(self, model, train_loader, val_loader, config):
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.config = config
        
    self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    self.loss_fn = DiffusionLoss()
        
def train(self):
    for epoch in range(self.config['epochs']):
        self.model.train()
        for images in self.train_loader:
            images = images.to(self.config['device'])
            output = self.model(images)
            loss = self.loss_fn(output, images)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
        print(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {loss.item()}")

        if (epoch + 1) % self.config['save_interval'] == 0:
            self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
    
 def save_model(self, path):
     torch.save(self.model.state_dict(), path)    
