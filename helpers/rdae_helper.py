from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.ae_backbone import CAE_backbone, AE_backbone
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from loss_functions.drae_loss import DRAELossAutograd
import numpy as np
from misc import AverageMeter


class RDAEHelper(TrainTestHelper):
    def __init__(self, n_channels, h, w, x_train, y_train, *args, **kwargs):
        super(RDAEHelper, self).__init__(*args, **kwargs)
        self.method_tag = "rdae"

        self.n_channels = n_channels
        flatten_size = 128
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist',
                                 'tinyimagenet', 'cifar10']:
            self.model = CAE_backbone(input_shape=(n_channels, h, w),
                                      hidden_layer_sizes=[32, 64, 128], bn=True,
                                      flatten_size=flatten_size).cuda()
        else:
            self.model = AE_backbone(input_shape=n_channels,
                                     hidden_layer_sizes=[512, 256, 128], bn=True,
                                     flatten_size=flatten_size).cuda(0)
        #self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = nn.MSELoss()
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = 250

        self.lmbda = 0.00065
        self.x_train = x_train
        self.y_train = y_train

        self.S = np.zeros_like(self.x_train)


    def train(self):
        self.losses = AverageMeter()
        test_in_epoch_target = self.test_in_epoch - 1

        def prox_l21(S, lmbda):
            """L21 proximal operator."""
            Snorm = np.sqrt((S ** 2).sum(axis=tuple(range(1, S.ndim)), keepdims=False))
            multiplier = 1 - 1 / np.minimum(Snorm / lmbda, 1)
            out = S * multiplier.reshape((S.shape[0],) + (1,) * (S.ndim - 1))
            return out

        def get_reconstruction(loader):
            self.model.eval()
            rc = []
            for inputs, _ in loader:
                with torch.no_grad():
                    rc.append(self.model(inputs.cuda()).cpu().numpy())
            out = np.concatenate(rc, axis=0)
            if len(out.shape) > 2:
                out = out.transpose((0, 2, 3, 1))
            return out

        for epoch in range(self.epochs):
            if len(self.x_train.shape) <= 2:
                trainset = trainset_pytorch(train_data=self.x_train - self.S, train_labels=self.y_train)
            else:
                trainset = trainset_pytorch(train_data=self.x_train - self.S, train_labels=self.y_train,
                                            transform=transform_train)
            trainloader = data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(trainloader):
                self.train_step(inputs, y)
                if batch_idx % self.test_in_epoch == test_in_epoch_target:
                    self.test(True)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                           self.losses.avg))

            testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
            recon = get_reconstruction(testloader)
            self.S = prox_l21(self.x_train - recon, self.lmbda)



    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        outputs = self.model(inputs)
        loss = self.criterion(inputs, outputs)

        # self.losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        losses = []
        y_test = []

        testset = trainset_pytorch(train_data=self.x_train - self.S, train_labels=self.y_train, transform=transform_test)
        if len(self.x_train.shape) <= 2:
            testset = trainset_pytorch(train_data=self.x_train - self.S, train_labels=self.y_train)
        self.testloader = data.DataLoader(testset, batch_size=1024, shuffle=False)
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            outputs = self.model(inputs)
            loss = outputs.sub(inputs).pow(2).view(outputs.size(0), -1)
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
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)