from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.LSA_mnist import LSA_MNIST
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
import numpy as np
from loss_functions import LSALoss



class LSAHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w,score_norm=True, *args, **kwargs):
        super(LSAHelper, self).__init__(*args, **kwargs)
        self.method_tag = "lsa"

        self.n_channels = n_channels
        cpd_channels = 100
        code_length = 64
        lam = 1.
        lr = 0.0001
        self.model = LSA_MNIST(input_shape=(n_channels, h, w),code_length=code_length,cpd_channels=cpd_channels).cuda()

        cudnn.benchmark = True
        self.criterion = LSALoss(cpd_channels=cpd_channels,lam=lam)
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=1e-6, lr=lr)
        #self.epochs = 250
        self.score_norm = score_norm
        self.print("score_norm:{} lam:{} lr:{}".format(score_norm, lam, lr))


    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r, z, z_dist = self.model(inputs)
        loss = self.criterion(inputs, x_r, z, z_dist)
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        #reloss = np.zeros(shape=len(self.testloader.dataset, ))
        reloss = np.zeros(shape=len(self.testloader.dataset, ))
        aurloss = np.zeros(shape=len(self.testloader.dataset, ))
        y_test = []
        cc = 0
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r, z, z_dist = self.model(x)
                loss = self.criterion(x, x_r, z, z_dist)
                #print(self.criterion.reconstruction_loss.shape)
                bs = len(inputs)
                reloss[cc:cc + bs] = -self.criterion.reconstruction_loss.cpu().numpy()
                aurloss[cc:cc + bs] = -self.criterion.autoregression_loss.cpu().numpy()
                #print(cc, cc+bs)
                cc += bs
                y_test.append(labels.data.cpu())
        y_test = torch.cat(y_test, dim=0)
        if self.score_norm:
            remax, remin = reloss.max(), reloss.min()
            aurmax, aurmin = aurloss.max(), aurloss.min()

            reloss = (reloss - remin) / (remax - remin + 1e-12)
            aurloss = (aurloss - aurmin) / (aurmax - aurmin + 1e-12)

        # scores are the sum of two loss
        #scores = reloss + aurloss
        scores = reloss + aurloss
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)