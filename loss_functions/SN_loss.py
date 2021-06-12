import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class L21Loss(nn.Module):
    def __init__(self):
        super(L21Loss, self).__init__()

    def forward(self, z):
        #x = x.view(x.shape[0], -1)
        #x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(z, p=2, dim=1)).mean()
        return le

class SNLossV1(nn.Module):
    def __init__(self, lamb1,noise_rate=0.1, iter_epoch=0, smooth_epoch = 0):
        super(SNLossV1, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate
        self.noise_rate_schedule = []
        for i in range(smooth_epoch):
            self.noise_rate_schedule += [self.noise_rate * i / smooth_epoch] * iter_epoch
        self.iter_count = 0
        self.mse = nn.MSELoss()
        self.l21loss = L21Loss()
        self.lamb1 = lamb1

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x - x_r, p=2, dim=1)).mean()
        return le

    def forward(self, x, x_r):
        return self.L21_error(x, x_r)

class MPCALossV1(nn.Module):
    def __init__(self, lamb1, lamb2, A, B, noise_rate, mode):
        super(MPCALossV1, self).__init__()
        self.l21loss = L21Loss()
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.A = A
        self.B = B
        self.z_size = A.shape[1]
        self.I = torch.eye(self.z_size).cuda()
        self.noise_rate = noise_rate
        self.mode = mode

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x - x_r, p=2, dim=1))
        return le

    def pca_loss(self, y, y_r):
        le = (torch.norm(y - y_r, p=2, dim=1))
        return le

    def proj_error(self):
        return (torch.matmul(self.A.t(), self.B) - self.I).pow(2).mean()

    def forward(self, x, x_r, y, y_r):
        l21 = self.L21_error(x, x_r)
        pca = self.pca_loss(y, y_r)
        proj = self.proj_error()
        if self.mode == 'none':
            pass
        elif self.mode == 'recons':
            idx = l21.argsort().detach().cpu().numpy()
            rem_num = int(x.size(0) * (1. - self.noise_rate))
            l21 = l21[idx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        elif self.mode == 'exchange':
            idx = l21.argsort().detach().cpu().numpy()
            zidx = pca.argsort().detach().cpu().numpy()
            rem_num = int(x.size(0) * (1. - self.noise_rate))
            l21 = l21[zidx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        else:
            raise NotImplementedError
        #print(l21.item(), pca.item(), proj.item())
        return l21 + self.lamb1 * pca + self.lamb2 * proj

class MPCALossV2(nn.Module):
    def __init__(self, lam, A, B, noise_rate, mode):
        super(MPCALossV2, self).__init__()
        self.l21loss = L21Loss()
        self.A = A
        self.B = B
        self.z_size = A.shape[1]
        self.I = torch.eye(self.z_size).cuda()
        self.noise_rate = noise_rate
        self.mode = mode
        self.lam = lam

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x - x_r, p=2, dim=1))
        return le

    def pca_loss(self, y, y_r):
        le = (torch.norm(y - y_r, p=2, dim=1))
        return le

    def proj_error(self):
        return (torch.matmul(self.A.t(), self.B) - self.I).pow(2).mean()

    def forward(self, x, x_r, y, y_r, step='ED'):
        l21 = self.L21_error(x, x_r)
        pca = self.pca_loss(y.detach(), y_r)
        rem_num = int(x.size(0) * (1. - self.noise_rate))
        #return l21.mean() #+ pca.mean()
        if step == 'ED':
            #zidx = pca.argsort().detach().cpu().numpy()
            #l21 = l21[zidx[:rem_num]].mean()
            #print(l21.item())
            return l21.mean()
        else:
            #proj = self.proj_error()
            idx = l21.argsort().detach().cpu().numpy()
            pca = pca[idx[:rem_num]].mean()
            #pca = pca.mean()
            #print(pca.item())
            return pca# + self.lam * proj


class MPCAGTLossV1(nn.Module):
    def __init__(self, lamb1, lamb2, A, B, noise_rate, mode):
        super(MPCAGTLossV1, self).__init__()
        self.l21loss = L21Loss()
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.A = A
        self.B = B
        self.z_size = A.shape[1]
        self.I = torch.eye(self.z_size).cuda()
        self.noise_rate = noise_rate
        self.mode = mode
        self.softmax = nn.Softmax(dim=1)

    def ce_error(self, x, labels):
        return F.cross_entropy(x, labels, reduction='none')

    def pca_loss(self, y, y_r):
        le = (torch.norm(y - y_r, p=2, dim=1))
        return le

    def proj_error(self):
        return (torch.matmul(self.A.t(), self.B) - self.I).pow(2).mean()

    def forward(self, o, labels, y, y_r):
        ce = self.ce_error(o, labels)
        pca = self.pca_loss(y, y_r)
        proj = self.proj_error()
        if self.mode == 'none':
            pass
        elif self.mode == 'recons':
            idx = ce.argsort().detach().cpu().numpy()
            rem_num = int(o.size(0) * (1. - self.noise_rate))
            ce = ce[idx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        elif self.mode == 'exchange':
            idx = ce.argsort().detach().cpu().numpy()
            zidx = pca.argsort().detach().cpu().numpy()
            rem_num = int(o.size(0) * (1. - self.noise_rate))
            ce = ce[zidx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        else:
            raise NotImplementedError
        #print(ce.item(), pca.item(), proj.item())
        return ce + self.lamb1 * pca + self.lamb2 * proj

    def neg_entropy(self, x):
        ne = self.softmax(x)
        return (ne *torch.log2(ne + 1e-12)).sum(dim=1)


class MPCAGTLossV2(nn.Module):
    def __init__(self, lamb1, lamb2, As, Bs, noise_rate, mode, batch_size, proj_mode):
        super(MPCAGTLossV2, self).__init__()
        self.l21loss = L21Loss()
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.As = As
        self.Bs = Bs
        self.z_size = As.shape[2]
        # self.I = torch.eye(self.z_size).cuda()
        # self.I = self.I.repeat(As.shape[0], 1, 1)
        self.proj_mode = proj_mode
        if proj_mode == 'batch':
            i = np.eye(self.z_size)
            i = np.expand_dims(i, axis=0)
            i = np.repeat(i, batch_size, axis=0)
            # print(i.shape)
            self.I = torch.from_numpy(i).cuda()
        else:
            i = np.eye(self.z_size)
            i = np.expand_dims(i, axis=0)
            i = np.repeat(i, As.shape[0], axis=0)
            self.I = torch.from_numpy(i).cuda()

        self.proj_mode = proj_mode
        self.noise_rate = noise_rate
        self.mode = mode
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size

    def ce_error(self, x, labels):
        return F.cross_entropy(x, labels, reduction='none')

    def pca_loss(self, y, y_r):
        le = (torch.norm(y - y_r, p=2, dim=1))
        return le

    def proj_error(self, labels):
        if self.proj_mode == 'batch':
            A = self.As[labels]
            B = self.Bs[labels]
            I = torch.eye(self.z_size).repeat(A.shape[0], 1, 1).cuda()
            return (torch.matmul(A.transpose(dim0=1, dim1=2), B) - I).pow(2).mean()
        else:
            return (torch.matmul(self.As.transpose(dim0=1,dim1=2), self.Bs) - self.I).pow(2).mean()
        # return (torch.matmul(self.As.transpose(dim0=1,dim1=2), self.As) - self.I).pow(2).mean() #TODO

    def forward(self, o, labels, y, y_r):
        ce = self.ce_error(o, labels)
        pca = self.pca_loss(y, y_r)
        proj = self.proj_error(labels)
        if self.mode == 'none':
            ce = ce.mean()
            pca = pca.mean()
        elif self.mode == 'recons':
            idx = ce.argsort().detach().cpu().numpy()
            rem_num = int(o.size(0) * (1. - self.noise_rate))
            ce = ce[idx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        elif self.mode == 'exchange':
            idx = ce.argsort().detach().cpu().numpy()
            zidx = pca.argsort().detach().cpu().numpy()
            rem_num = int(o.size(0) * (1. - self.noise_rate))
            ce = ce[zidx[:rem_num]].mean()
            pca = pca[idx[:rem_num]].mean()
        elif self.mode == 'union':
            idx = ce.argsort().detach().cpu().numpy()
            #zidx = pca.argsort().detach().cpu().numpy()
            rem_num = int(o.size(0) * (1. - self.noise_rate))
            #union_idx = np.union1d(idx[:rem_num], zidx[:rem_num])
            ce = ce[idx[:rem_num]].mean()
            pca = pca.mean()
        else:
            raise NotImplementedError
        #print(ce.item(), pca.item(), proj.item())
        return ce + self.lamb1 * pca + self.lamb2 * proj

    def neg_entropy(self, x):
        ne = self.softmax(x)
        return (ne *torch.log2(ne + 1e-12)).sum(dim=1)

    def pl_mean(self, x, labels):
        #print(x.shape, labels.shape)
        return -F.cross_entropy(x, labels, reduction='none')