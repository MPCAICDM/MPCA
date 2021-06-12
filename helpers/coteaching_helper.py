from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.LSA_mnist import LSA_MNIST
from models.LSA_cifar10 import LSACIFAR10
from models.encoders_decoders import CAE_pytorch
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch, transformer_dataset
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
import numpy as np
from misc import AverageMeter
from transformations import RA, RA_IA, RA_IA_PR, Rotate4D
from models.wrn_pytorch import WideResNet
from models.resnet_pytorch import ResNet
from models.densenet_pytorch import DenseNet
from loss_functions.coteaching_loss import CoTeachingLoss, CoTeachingResnetLoss

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score)

class CoTeachingResnetHelper(TrainTestHelper):
    def __init__(self, n_channels,OP_TYPE,BACKEND,SCORE_MODE, *args, **kwargs):
        super(CoTeachingResnetHelper, self).__init__(*args, **kwargs)
        self.method_tag = "coteaching_resnet"

        self.n_channels = n_channels
        self.SCORE_MODE = SCORE_MODE
        self.OP_TYPE = OP_TYPE
        if OP_TYPE == 'RA':
            transformer = RA(8, 8)
        elif OP_TYPE == 'RA+IA':
            transformer = RA_IA(8, 8, 12)
        elif OP_TYPE == 'RA+IA+PR':
            transformer = RA_IA_PR(8, 8, 12, 23, 2)
        elif OP_TYPE == 'Rotate4D':
            transformer = Rotate4D()
        else:
            raise NotImplementedError
        print(OP_TYPE)
        print(transformer.n_transforms)
        self.transformer = transformer

        self.BACKEND = BACKEND
        if BACKEND == 'wrn':
            n, k = (10, 4)
            self.model1 = WideResNet(num_classes=transformer.n_transforms, depth=n, widen_factor=k, in_channel=n_channels).cuda()
            self.model2 = WideResNet(num_classes=transformer.n_transforms, depth=n, widen_factor=k, in_channel=n_channels).cuda()
        else:
            raise NotImplementedError('Unimplemented backend: {}'.format(BACKEND))
        print('Using backend: {} ({})'.format(type(self.model1).__name__, BACKEND))

        self.num_workers = 8
        self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = CoTeachingResnetLoss(noise_rate=0.1, score_mode=SCORE_MODE)
        # use adam always
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            self.optimizer1 = optim.SGD(self.model1.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
            self.optimizer2 = optim.SGD(self.model2.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        else:
            self.optimizer1 = optim.Adam(self.model1.parameters(), eps=1e-7, weight_decay=0.0005)
            self.optimizer2 = optim.Adam(self.model2.parameters(), eps=1e-7, weight_decay=0.0005)
        self.epochs = int(np.ceil(250 / transformer.n_transforms))

    def transform_traindata(self, x_train):
        print('transform_traindata')
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train))
        #self.x_train_task_transformed = self.transformer.transform_batch(
        #    np.repeat(x_train, self.transformer.n_transforms, axis=0), transformations_inds)
        self.trainset = transformer_dataset(train_data=x_train,
                                         train_labels=transformations_inds,
                                            data_transformer=self.transformer,
                                            transform=transform_train)
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
        #self.test_in_epoch = len(self.trainloader) * 6 // (self.transformer.n_transforms)
        #print(self.test_in_epoch)

    def transform_testdata(self, x_test, y_test):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_test))
        #self.x_test_task_transformed = self.transformer.transform_batch(
        #    np.repeat(x_test, self.transformer.n_transforms, axis=0), transformations_inds)
        self.testset = transformer_dataset(train_data=x_test,
                                         train_labels=transformations_inds,
                                           data_transformer=self.transformer,
                                        transform=transform_test)
        self.y_test = y_test
        #self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
        self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.num_workers)

    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model1.train()
            self.model2.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        inputs, targets = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        outputs1, _ = self.model1(inputs)
        outputs2, _ = self.model2(inputs)
        loss1, loss2 = self.criterion(outputs1, outputs2, targets)

        self.losses.update(loss1.item(), 1)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

    def test_step(self, x, y):
        y = y.view(x.size(0), -1)
        targets = []
        for i in range(y.size(1)):
            targets.append(y[:, i].cuda())
        inputs = torch.autograd.Variable(x.cuda())
        outputs, _ = self.model1(inputs)
        scores = self.criterion.predict(outputs, targets)
        return scores

    def compute_scores(self):
        res = []
        self.model1.eval()
        with torch.no_grad():
            for batch_idx, (inputs, y) in enumerate(self.testloader):
                inputs = torch.autograd.Variable(inputs.cuda())
                scores = self.test_step(inputs, y)
                res.append(scores)
        res = torch.cat(res, dim=0)
        preds = res.view(len(self.y_test), self.transformer.n_transforms)
        preds = preds.mean(dim=1)
        return preds.cpu().numpy(), self.y_test

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)


class CoTeachingHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w, *args, **kwargs):
        super(CoTeachingHelper, self).__init__(*args, **kwargs)
        self.method_tag = "coteaching"

        self.n_channels = n_channels
        cpd_channels = 10
        code_length = 64
        lam = 1
        self.model1 = CAE_pytorch(in_channels=self.n_channels).cuda()
        self.model2 = CAE_pytorch(in_channels=self.n_channels).cuda()
        '''
        if self.dataset_name == 'mnist':
            self.model1 = LSA_MNIST(input_shape=(n_channels, h, w),code_length=code_length,cpd_channels=cpd_channels).cuda()
            self.model2 = LSA_MNIST(input_shape=(n_channels, h, w),code_length=code_length,cpd_channels=cpd_channels).cuda()
        else:
            self.model1 = LSACIFAR10(input_shape=(n_channels, h, w), code_length=code_length,
                                    cpd_channels=cpd_channels).cuda()
            self.model2 = LSACIFAR10(input_shape=(n_channels, h, w), code_length=code_length,
                                    cpd_channels=cpd_channels).cuda()
        '''
        self.batch_size = 128


        cudnn.benchmark = True
        self.criterion = CoTeachingLoss(noise_rate=0.1)
        # use adam always
        self.optimizer1 = optim.Adam(self.model1.parameters(), eps=1e-7, weight_decay=0.0005)
        self.optimizer2 = optim.Adam(self.model2.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = 250

    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model1.train()
            self.model2.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r1= self.model1(inputs)
        x_r2 = self.model2(inputs)
        loss1, loss2 = self.criterion(x_r1, x_r2, inputs)

        #self.losses.update(loss1.item(), inputs.size(0))
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

    def compute_scores(self):
        self.model1.eval()
        #reloss = np.zeros(shape=len(self.testloader.dataset, ))
        losses = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r = self.model1(x)
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

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)