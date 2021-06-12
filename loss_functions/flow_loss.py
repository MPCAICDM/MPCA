import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from loss_functions.reconstruction_loss import ReconstructionLoss

#JJ: rewrite Autoregression Loss
class FlowLoss(nn.Module):
    """
    T(s) = z s~N(0,1), z~q
    loss = -mean( log q(z_i)), average loss in every batch
    """
    def __init__(self):
        super(FlowLoss, self).__init__()

    def forward(self, s, log_jacob,size_average=True):
        '''
        Args:
            s, source data s~ N(0,1) T(s) = z
            log_jacob: log of jacobian of T-inverse

         
        return: the mean negative log-likelihood (averaged along the batch axis)
        '''
        s_d = s
        log_jacob_d = log_jacob
        
        log_probs = (-0.5 * s_d.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        
        # formula (3)
        loss = -(log_probs + log_jacob_d).sum(-1, keepdim=True)
        log_probs = log_probs.sum(-1,keepdim=True)
        log_jacob_d = log_jacob_d.sum(-1,keepdim=True)

        if size_average:
            loss = loss.mean()
            log_probs= log_probs.mean()
            log_jacob_d = log_jacob_d.mean()
        else:
            loss = loss.squeeze(-1)
            log_probs= log_probs.squeeze(-1)
            log_jacob_d = log_jacob_d.squeeze(-1)

        return loss, -log_probs,-log_jacob_d


class MTQSOSLoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """

    def __init__(self, lam=1):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(MTQSOSLoss, self).__init__()

        self.lam = lam

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = FlowLoss()

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None

        self.total_loss = None
        # self.nlog_probs = None
        # self.nagtive_log_jacob = None

    def forward(self, x, x_r, s, nagtive_log_jacob, average=True):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r, average)
        arg_loss, nlog_probs, nlog_jacob_d = self.autoregression_loss_fn(s, nagtive_log_jacob, average)

        tot_loss = rec_loss + self.lam * arg_loss

        # Store numerical
        self.reconstruction_loss = rec_loss
        self.autoregression_loss = arg_loss

        # self.nlog_probs = nlog_probs * self.lam
        # self.nagtive_log_jacob = nlog_jacob_d * self.lam

        self.total_loss = tot_loss

        return tot_loss
