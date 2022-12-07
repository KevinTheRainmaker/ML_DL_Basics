import torch
import torch.nn as nn
import torch.nn.functional as F


class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(
            torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(
            torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(
            torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(
            torch.zeros(dims_in), requires_grad=True)
        self.eps = eps

    def forward(self, x, mask):
        # fill the blank
        mask = F.interpolate(mask.detach(), size=x.size()[2:])  # resized_mask

        fg_mean, fg_std = self.get_foreground_mean_std(x * mask, mask)
        fg_norm = (self.foreground_gamma[None, :, None, None] * (
            (x - fg_mean) / fg_std)) + self.foreground_beta[None, :, None, None]

        fg_norm = fg_norm * \
            self.background_gamma[None, :, None, None] + \
            self.background_beta[None, :, None, None]

        normalized_foreground = fg_norm * mask

        bg_mask = (1-mask)
        bg_mean, bg_std = self.get_foreground_mean_std(x * bg_mask, bg_mask)
        bg_norm = (self.background_gamma[None, :, None, None]*(
            (x - bg_mean) / bg_std)) + self.background_beta[None, :, None, None]

        normalized_background = bg_norm * mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean)
                        ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        # fill the blank
        # num = torch.sum(mask, dim=[2, 3])
        # sigma = torch.sum(region, dim=[2, 3])

        # mean = (sigma / num)[:, :, None, None]
        # var = torch.sum(region-mean, dim=[2, 3])/num

        return mean, torch.sqrt(var+self.eps)
