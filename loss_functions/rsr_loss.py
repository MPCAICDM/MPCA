import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class RSRLoss(nn.Module):
    def __init__(self, lamb1, lamb2, A, noise_rate):
        super(RSRLoss, self).__init__()
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.A = A
        self.z_size = A.shape[1]
        self.I = torch.eye(self.z_size).cuda()
        self.noise_rate = noise_rate

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x-x_r, p=2, dim=1))
        return le

    def pca_error(self, y, z):
        z = torch.matmul(z, self.A.t())
        return (torch.norm(y-z, p=2, dim=1))

    def proj_error(self):
        return (torch.matmul(self.A.t(), self.A) - self.I).pow(2).mean()

    def forward(self, x, x_r, y, y_rsr):
        l21err = self.L21_error(x, x_r)
        l21sort = l21err.argsort().detach().cpu().numpy()
        pcaerr = self.pca_error(y, y_rsr)
        pcasort = pcaerr.argsort().detach().cpu().numpy()
        rem_num = int(x.size(0) * (1. - self.noise_rate))
        l21err = l21err[pcasort[:rem_num]].mean()
        pcaerr = pcaerr[l21sort[:rem_num]].mean()
        proj_err = self.proj_error()
        #print(rem_num)
        #print(l21err.item(), pcaerr.item(), proj_err.item())
        return l21err + self.lamb1 * pcaerr + self.lamb2 * proj_err

