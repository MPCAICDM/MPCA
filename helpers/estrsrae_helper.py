from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data, get_class_name_from_index
from models.LSA_mnist import LSA_MNIST, LSA_MNIST_DOUBLE
from models.encoders_decoders import CAE_group_pytorch
from keras2pytorch_dataset import trainset_pytorch, transformer_dataset
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from transformations import RA, RA_IA, RA_IA_PR, Rotate4D
from models.wrn_pytorch import WideResNet
from models.resnet_pytorch import ResNet
from models.densenet_pytorch import DenseNet
from models.SN_mnist import SN_MNIST, RSRBoneV2, RSRBoneV2Linear, RSRBoneV3Linear
import numpy as np
from misc import AverageMeter
from loss_functions.coteaching_loss import InCoTeachingEstLoss
import itertools
import os

class EstRSRAEHelper(TrainTestHelper):
    def __init__(self, n_channels, noise_rate, group=2, lamb1=1., mode='exchange', *args, **kwargs):
        super(EstRSRAEHelper, self).__init__(*args, **kwargs)
        self.method_tag = "EstRSRAE"

        self.n_channels = n_channels
        self.group = group
        # self.print("group: {}".format(group))
        # self.model = LSA_MNIST_DOUBLE(input_shape=(n_channels, h, w),code_length=code_length,
        #                              cpd_channels=cpd_channels, group=group).cuda()
        # self.model = CAE_group_pytorch(in_channels=self.n_channels, group=self.group).cuda()
        lr = 0.00025
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist']:
            pass
        else:
            self.model = RSRBoneV3Linear(input_shape=n_channels, z_channels=10,
                                         hidden_layer_sizes=[32, 64, 128], bn=False).cuda()
            # lr = 0.001

        self.lamb1 = lamb1
        # self.batch_size = 128
        # self.print(noise_rate)


        self.criterion = InCoTeachingEstLoss(lamd=self.lamb1, noise_rate=noise_rate, cpd_channels=100,
                                             mode=mode).cuda()
        cudnn.benchmark = True

        self.noise_rate = noise_rate
        self.print("lamb1:{}  noise_rate: {} mode:{}".format(self.lamb1, noise_rate, mode))
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-7)
        # self.epochs = 250

    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            # if self.criterion.noise_rate < self.noise_rate:
            #    self.criterion.noise_rate += 0.05
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                           self.losses.avg))

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        z, x_r, z_dist = self.model(inputs)
        loss, _, _ = self.criterion(x_r, inputs, z, z_dist)
        self.losses.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        scoreRes = []
        scoreAug = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            # print(x)
            with torch.no_grad():
                z, x_r, z_dist = self.model(x)
                _, mse, autoreg = self.criterion(x_r, x, z, z_dist)
                mse = mse.pow(2)
                scoreRes.append(-mse.cpu())
                scoreAug.append(-autoreg.cpu())

                y_test.append(labels.data.cpu())


        scoreRes = torch.cat(scoreRes, dim=0)
        scoreAug = torch.cat(scoreAug, dim=0)
        y_test = torch.cat(y_test, dim=0)
        remin, remax = scoreRes.min(), scoreRes.max()
        augmin, augmax = scoreAug.min(), scoreAug.max()
        scoreRes = (scoreRes - remin) / (remax - remin + 1e-12)
        scoreAug = (scoreAug - augmin) / (augmax - augmin + 1e-12)
        return (scoreRes, scoreAug, scoreRes + scoreAug), y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            tags = ['Reconstruction', 'autoregress', 'Combine']
            for i in range(len(tags)):
                self.print('score {}'.format(tags[i]))
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)