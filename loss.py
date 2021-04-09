import torch.nn as nn
import torch
import torchvision
import pytorch_msssim as py_ms
from math import log10


class supervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16_conv_4_3 = nn.Sequential(
            *list(torchvision.models.vgg16(pretrained=True).children())[0][:22]
        )
        self.L1_lossFn = nn.L1_lossFn
        self.MSE_LossFn = nn.MSE_LossFn

    def __call__(
        self,
        Ft_p,
        IFrame,
        g_I0_F_t_0,
        g_I1_F_t_1,
        valFlowBackWarp_I0_F_1_0_I_1,
        valFlowBackWarp_I1_F_0_1_I_0,
        F_1_0,
        F_0_1,
    ):
        recnLoss = self.L1_lossFn(Ft_p, IFrame)
        prcpLoss = self.MSE_LossFn(
            self.vgg16_conv_4_3(Ft_p), self.vgg16_conv_4_3(IFrame)
        )
        warpLoss = (
            self.L1_lossFn(g_I0_F_t_0, IFrame)
            + self.L1_lossFn(g_I1_F_t_1, IFrame)
            + valFlowBackWarp_I0_F_1_0_I_1
            + valFlowBackWarp_I1_F_0_1_I_0
        )
        loss_smooth = (
            torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:]))
            + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            + torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:]))
            + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        )
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
        return loss


def psnr(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return 10 * log10(1 / nn.MSE_LossFn(Ft_p, IFrame))
    else:
        return 10 * log10(1 / nn.MSE_LossFn(Ft_p, IFrame).item())


def ssim(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True)
    else:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True).item()
