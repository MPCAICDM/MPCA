import torch
import torch.nn as nn
import torch.nn.functional as F

class E3outlierOELoss(nn.Module):
    def __init__(self,lamb, size_average=True):
        super(E3outlierOELoss, self).__init__()
        self.lamb = lamb
        self.size_average = size_average
        #print('E3outlierOELoss')

    def forward(self, x, targets, aux=None):
        err = F.cross_entropy(x, targets,reduction='none')
        #err_sorted, _ = torch.sort(err)
        uni_err = -(x.mean(1) - torch.logsumexp(x, dim=1))
        uni_err, _ = torch.sort(uni_err)
        spl = int(x.size(0) * (self.lamb))
        loss = err.mean() + self.lamb * uni_err[:spl].mean() # wrong implements
        return loss

'''
ce = F.cross_entropy(x, targets,reduction='none')
        uniform = -(x.mean(1) - torch.logsumexp(x, dim=1))


        #err_sorted, _ = torch.sort(err)
        #spl = int(x.size(0) * (1 - self.lamb))
        num_norm = aux_labels.sum()
        loss =  (ce * aux_labels).sum()/num_norm + (uniform * (1 - aux_labels)).sum() / (len(aux_labels)-num_norm)
        return loss

'''