from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.LSA_mnist import LSA_MNIST
from models.LSA_cifar10 import LSACIFAR10
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
import numpy as np



class CAELSAHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w, *args, **kwargs):
        super(CAELSAHelper, self).__init__(*args, **kwargs)
        self.method_tag = "cae_lsa"

        self.n_channels = n_channels
        cpd_channels = 10
        code_length = 10
        lam = 1
        if self.dataset_name == 'cifar10':
            self.model = LSACIFAR10(input_shape=(n_channels, h, w), code_length=code_length,
                                    cpd_channels=cpd_channels).cuda()
        else:
            self.model = LSA_MNIST(input_shape=(n_channels, h, w), code_length=code_length,
                                   cpd_channels=cpd_channels).cuda()
            print('LSA_MNIST')
        self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = nn.MSELoss(size_average=False)
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = 250


    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r, z, _ = self.model(inputs)
        loss = self.criterion(inputs, x_r)
        #le = (torch.norm(z, p=2, dim=1)).mean()
        #loss += le
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        #reloss = np.zeros(shape=len(self.testloader.dataset, ))
        losses = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r, z, z_dist = self.model(x)
                loss = x_r.sub(x).pow(2).view(x_r.size(0), -1)
                loss = loss.sum(dim=1, keepdim=False)
                losses.append(loss.data.cpu())
                y_test.append(labels.data.cpu())
        losses = torch.cat(losses, dim=0)
        y_test = torch.cat(y_test, dim=0)
        losses = losses.numpy()
        losses = losses - losses.min()
        losses = losses / (1e-8 + losses.max())
        scores = 1 - losses
        return scores, y_test.numpy()

    def compute_scores_hidden(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        losses = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r, y, z_dist = self.model(x)
                loss = y.pow(2).view(y.size(0), -1)
                loss = loss.sum(dim=1, keepdim=False)
                losses.append(loss.data.cpu())
                y_test.append(labels.data.cpu())
        losses = torch.cat(losses, dim=0)
        y_test = torch.cat(y_test, dim=0)
        losses = losses.numpy()
        losses = losses - losses.min()
        losses = losses / (1e-8 + losses.max())
        scores = 1 - losses
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
            '''
            scoresB, _ = self.compute_scores_hidden()
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scoresB, y_test)
            self.print('score Hidden')
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

            scores = scores + scoresB
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print('score Combine')
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
            '''
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)