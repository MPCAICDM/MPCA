import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class CoTeachingLoss(nn.Module):
    def __init__(self, noise_rate=0.1):
        super(CoTeachingLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate

    def forward(self, xr1, xr2, x):
        mse1 = self.mse(xr1, x).mean(dim=(1,2,3))
        mse2 = self.mse(xr2, x).mean(dim=(1,2,3))
        idxsortd1 = mse1.argsort().detach()
        idxsortd2 = mse2.argsort().detach()
        #idxsortd1 = np.argsort(mse1.cpu().data)
        #idxsortd2 = np.argsort(mse2.cpu().data)
        #print(idxsortd1)
        #print(mse1,mse2,idxsortd1,idxsortd2)
        #return mse1.mean(), mse2.mean()
        rem_num = int(x.size(0) * (1. - self.noise_rate))
        return mse1[idxsortd2[:rem_num]].mean(), \
               mse2[idxsortd1[:rem_num]].mean()

class CoTeachingResnetLoss(nn.Module):
    def __init__(self, noise_rate=0.1, score_mode='pl_mean'):
        super(CoTeachingResnetLoss, self).__init__()
        self.noise_rate = noise_rate
        self.score_mode = score_mode
        self.softmax = nn.Softmax(dim=1)

    def forward(self, o1, o2, labels):
        ce1 = F.cross_entropy(o1, labels, reduction='none')
        ce2 = F.cross_entropy(o2, labels, reduction='none')
        idxsortd1 = ce1.argsort().detach()
        idxsortd2 = ce2.argsort().detach()
        rem_num = int(o1.size(0) * (1. - self.noise_rate))
        return ce1[idxsortd2[:rem_num]].mean(), \
               ce2[idxsortd1[:rem_num]].mean()

    def pl_mean(self, x, labels):
        raise NotImplementedError

    def neg_entropy(self, x):
        ne = self.softmax(x)
        return (ne *torch.log2(ne)).sum(dim=1)

    def predict(self, x, labels):
        if self.score_mode == 'pl_mean':
            return self.pl_mean(x, labels)
        else:
            return self.neg_entropy(x)

class InCoTeachingEstLoss(nn.Module):
    def __init__(self, lamd, cpd_channels, mode, noise_rate=0.1):
        super(InCoTeachingEstLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate
        self.lamd = lamd
        #self.lamd2 = 0.05
        self.coteach_mode = mode # 'exchange' 'union' 'intersect'


        self.cpd_channels = cpd_channels

        # Avoid nans
        self.eps = np.finfo(float).eps

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x - x_r, p=2, dim=1)).mean()
        return le

    def Autoregress_error(self, z, z_dist):
        z_d = z.detach()

        # Apply softmax
        z_dist = F.softmax(z_dist, dim=1)

        # Flatten out codes and distributions
        z_d = z_d.view(len(z_d), -1).contiguous()
        z_dist = z_dist.view(len(z_d), self.cpd_channels, -1).contiguous()

        # Log (regularized), pick the right ones
        z_dist = torch.clamp(z_dist, self.eps, 1 - self.eps)
        log_z_dist = torch.log(z_dist)
        index = torch.clamp(torch.unsqueeze(z_d, dim=1) * self.cpd_channels, min=0,
                            max=(self.cpd_channels - 1)).long()
        selected = torch.gather(log_z_dist, dim=1, index=index)
        selected = torch.squeeze(selected, dim=1)

        # Sum and mean
        S = torch.sum(selected, dim=-1)
        nll = -S

        return nll

    def forward(self, xr, x, z, z_dist):
        x = x.view(x.shape[0], -1)
        xr = xr.view(xr.shape[0], -1)
        lmse = torch.norm(x - xr, p=2, dim=1)
        idxsorted = lmse.argsort().detach().cpu().numpy()

        rem_num = int(x.size(0) * (1. - self.noise_rate))
        arg_err = self.Autoregress_error(z, z_dist)
        zidxsorted = arg_err.argsort().detach().cpu().numpy()
        if self.coteach_mode == 'exchange':
            a = lmse[zidxsorted[:rem_num]].mean()
            b = arg_err[idxsorted[:rem_num]].mean()
            #c = arg_err[idxsorted[rem_num:]].mean()
            loss =  a + b * self.lamd
        elif self.coteach_mode == 'neg':
            a = lmse[zidxsorted[:rem_num]].mean()
            b = arg_err[idxsorted[:rem_num]].mean()
            c = arg_err[idxsorted[rem_num:]].mean()
            loss = a + (b - c) * self.lamd
        else:
            loss = lmse.mean() + self.lamd * arg_err.mean()
            #print(a.item(), b.item())
        return loss, lmse, arg_err


class InCoTeachingHiddenLoss(nn.Module):
    def __init__(self, lamd, noise_rate=0.1, group=2):
        super(InCoTeachingHiddenLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate
        self.group = group
        self.lamd = lamd
        self.lamd2 = 0.05

    def L21_error(self, x, x_r):
        x = x.view(x.shape[0], -1)
        x_r = x_r.view(x_r.shape[0], -1)
        le = (torch.norm(x - x_r, p=2, dim=1)).mean()
        return le

    def forward(self, xr, x, z):
        L_mse = []
        idxsorted = []
        x = x.view(x.shape[0], -1)
        #print(len(xr), xr[0].shape, x.shape, z.shape)
        for ixr in xr:
            ixr = ixr.view(ixr.shape[0], -1)
            #print(x.shape, ixr.shape)
            #lmse = self.mse(ixr, x).mean(dim=1)
            lmse = torch.norm(x - ixr, p=2, dim=1)
            #lmse = ixr.sub(x).pow(2).view(ixr.size(0), -1).sum(dim=1, keepdim=False)
            L_mse.append(lmse)
            idxsorted.append(lmse.argsort().detach().cpu().numpy())
        rem_num = int(x.size(0) * (1. - self.noise_rate))
        znorm = torch.norm(z, p=2, dim=1)
        zidxsorted = znorm.argsort().detach().cpu().numpy()
        shift = random.randint(0, self.group - 1)
        loss = 0
        #print(rem_num)
        for i in range(self.group):
            loss += L_mse[i][zidxsorted[:rem_num]].mean()
            #loss +=  L_mse[i].mean()

        return znorm[idxsorted[(shift)%self.group][:rem_num]].mean() * self.lamd + loss
        #return znorm.mean()+loss

class InCoTeachingAgreeLoss(nn.Module):
    def __init__(self, noise_rate=0.1, group=2):
        super(InCoTeachingAgreeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate
        self.group = group

    def forward(self, xr, x):
        L_mse = []
        idxsorted = []
        for ixr in xr:
            lmse = self.mse(ixr, x).mean(dim=(1,2,3))
            L_mse.append(lmse)
            idxsorted.append(lmse.argsort().detach().cpu().numpy())
        rem_idx = int(x.size(0) * (1. - self.noise_rate))
        loss = 0
        agrees = idxsorted[0][:rem_idx]
        for i in range(1, self.group):
            agrees = np.intersect1d(agrees,idxsorted[i])

        for i in range(self.group):
            loss += L_mse[i][agrees].mean()
            #loss += L_mse[i].mean()
        return loss


class InCoTeachingLoss(nn.Module):
    def __init__(self, noise_rate=0.1, group=2):
        super(InCoTeachingLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.noise_rate = noise_rate
        self.group = group

    def forward(self, xr, x):
        L_mse = []
        idxsorted = []
        for ixr in xr:
            lmse = self.mse(ixr, x).mean(dim=(1,2,3))
            L_mse.append(lmse)
            idxsorted.append(lmse.argsort().detach())
        rem_num = int(x.size(0) * (1. - self.noise_rate))
        shift = random.randint(1, self.group - 1)
        loss = 0
        for i in range(self.group):
            loss += L_mse[i][idxsorted[(i+shift)%self.group][:rem_num]].mean()
            #loss += L_mse[i].mean()
        return loss

class MulCEInCoTeachingLoss(nn.Module):
    def __init__(self, noise_rate=0.1, group=(2, 3, 3, 4), score_mode='pl_mean',
                 iter_per_epoch=100, smooth_epoch=0, oe_scale=None,
                 mask_group=None):
        super(MulCEInCoTeachingLoss, self).__init__()
        self.noise_rate = noise_rate
        self.group = group
        self.gsize = len(group)
        self.softmax = nn.Softmax(dim=1)
        self.score_mode = score_mode
        self.noise_rate_schedule = []
        for i in range(smooth_epoch):
            self.noise_rate_schedule += [self.noise_rate * i / smooth_epoch] * iter_per_epoch
        self.iter_count = 0
        self.oe_scale = oe_scale

        if mask_group is None:
            self.mask_group = [False] * len(self.group)
        else:
            self.mask_group = mask_group

    def get_noist_rate(self):
        if self.iter_count < len(self.noise_rate_schedule):
            ns = self.noise_rate_schedule[self.iter_count]
        else:
            ns = self.noise_rate
        self.iter_count += 1
        return ns

    def forward(self, x, labels):
        noise_rate = self.get_noist_rate()
        Lce = []
        now = 0
        idxsorted = []
        for i in range(len(self.group)):
            if not self.mask_group[i]:
                lce = F.cross_entropy(x[:, now: now+self.group[i]], labels[i], reduction='none')
                Lce.append(lce)
                idxsorted.append(lce.argsort().detach())
            now += self.group[i]
        #print(now)
        rem_num = int(x.size(0) * (1. - noise_rate))
        shift = random.randint(0, len(Lce) - 1)
        loss = 0
        for i in range(len(Lce)):
            loss += Lce[i][idxsorted[(i+shift)%len(Lce)][:rem_num]].mean()
            #loss += Lce[i].mean()

        if self.oe_scale is not None:
            oe_num = -int(x.size(0) * noise_rate * self.oe_scale)
            now = 0
            for i in range(len(self.group)):
                if not self.mask_group[i]:
                    xi = x[idxsorted[(i + shift) % len(Lce)][oe_num:], now:now + self.group[i]]
                    loss += -0.1 * (xi.mean(dim=1) - torch.logsumexp(xi, dim=1)).mean()
                now += self.group[i]
        return loss

    def pl_mean(self, x, labels):
        Lce = []
        now = 0
        #print(x.shape, labels[0].shape)
        for i in range(len(self.group)):
            if not self.mask_group[i]:
                lce = -F.cross_entropy(x[:, now: now + self.group[i]], labels[i], reduction='none')
                Lce.append(lce)
            now += self.group[i]
        loss = 0
        for i in range(len(Lce)):
            loss += Lce[i]
        return loss

    def neg_entropy(self, x):
        neg_entropy = 0
        now = 0
        for i in range(len(self.group)):
            if not self.mask_group[i]:
                ne = self.softmax(x[:, now: now+self.group[i]])
                ne = ne * torch.log2(ne)
                ne = ne.sum(dim=1)
                neg_entropy += ne
            now += self.group[i]
        return neg_entropy

    def predict(self, x, labels):
        if self.score_mode == 'pl_mean':
            return self.pl_mean(x, labels)
        else:
            return self.neg_entropy(x)