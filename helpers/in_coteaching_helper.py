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
from models.SN_mnist import SN_MNIST, RSRBoneV2, RSRBoneV2Linear
import numpy as np
from misc import AverageMeter
from loss_functions.coteaching_loss import CoTeachingLoss, InCoTeachingLoss, MulCEInCoTeachingLoss,\
    InCoTeachingAgreeLoss,InCoTeachingHiddenLoss
import itertools
import os

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score)


class InCoTeachingAgreeHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w,noise_rate, group=2, *args, **kwargs):
        super(InCoTeachingAgreeHelper, self).__init__(*args, **kwargs)
        self.method_tag = "incoteaching-agree"

        self.n_channels = n_channels
        lam = 1
        self.group = group
        self.print("group: {}".format(group))
        #self.model = LSA_MNIST_DOUBLE(input_shape=(n_channels, h, w),code_length=code_length,
        #                              cpd_channels=cpd_channels, group=group).cuda()
        self.model = CAE_group_pytorch(in_channels=self.n_channels, group=self.group).cuda()
        self.batch_size = 128


        cudnn.benchmark = True
        self.criterion = InCoTeachingAgreeLoss(noise_rate=noise_rate, group=group)
        self.noise_rate = noise_rate
        self.print("noise_rate: {}".format(noise_rate))
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = 250


    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r, _ = self.model(inputs)
        loss = self.criterion(x_r, inputs)
        self.losses.update(loss.item(), 1)
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
                _, x_r = self.model(x)
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

class InCoTeachingHiddenHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w,noise_rate, add_conv,group=2,lamb1=1., *args, **kwargs):
        super(InCoTeachingHiddenHelper, self).__init__(*args, **kwargs)
        self.method_tag = "incoteaching"

        self.n_channels = n_channels
        self.group = group
        #self.print("group: {}".format(group))
        #self.model = LSA_MNIST_DOUBLE(input_shape=(n_channels, h, w),code_length=code_length,
        #                              cpd_channels=cpd_channels, group=group).cuda()
        #self.model = CAE_group_pytorch(in_channels=self.n_channels, group=self.group).cuda()
        lr = 0.00025
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist']:
            self.model = RSRBoneV2(input_shape=(n_channels, h, w), z_channels=30,
                                   hidden_layer_sizes=[32, 64, 128], group=group, add_conv=add_conv).cuda()
        else:
            self.model = RSRBoneV2Linear(input_shape=n_channels, z_channels=10,
                                         hidden_layer_sizes=[32, 64, 128],group=1, bn=False).cuda()
            #lr = 0.001

        self.lamb1 = lamb1
        #self.batch_size = 128
        #self.print(noise_rate)

        self.criterion = InCoTeachingHiddenLoss(lamd=self.lamb1,noise_rate=noise_rate, group=group).cuda()
        cudnn.benchmark = True

        self.noise_rate = noise_rate
        self.print("lamb1:{} add_conv:{} noise_rate: {} group:{}".format(self.lamb1, add_conv, noise_rate, group))
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=1e-6, lr=lr)
        #self.epochs = 250


    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            #if self.criterion.noise_rate < self.noise_rate:
            #    self.criterion.noise_rate += 0.05
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        z, x_r, _= self.model(inputs)
        #loss = self.criterion(x_r, inputs)
        loss = self.criterion(x_r, inputs, z)
        self.losses.update(loss.item(), 1)
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
            #print(x)
            with torch.no_grad():
                _, _, x_r = self.model(x)
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
                y, _, _ = self.model(x)
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
            self.print('score Reconstruction')
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

            scoresB, _ = self.compute_scores_hidden()
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scoresB, y_test)
            self.print('score Hidden')
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

            scores = scores + scoresB
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print('score Combine')
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

class InCoTeachingHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w,noise_rate, group=2, *args, **kwargs):
        super(InCoTeachingHelper, self).__init__(*args, **kwargs)
        self.method_tag = "incoteaching-hidden"

        self.n_channels = n_channels
        lam = 1
        self.group = group
        self.print("group: {}".format(group))
        #self.model = LSA_MNIST_DOUBLE(input_shape=(n_channels, h, w),code_length=code_length,
        #                              cpd_channels=cpd_channels, group=group).cuda()
        self.model = CAE_group_pytorch(in_channels=self.n_channels, group=self.group).cuda()
        self.batch_size = 128


        cudnn.benchmark = True
        self.criterion = InCoTeachingLoss(noise_rate=noise_rate, group=group)
        self.noise_rate = noise_rate
        self.print("noise_rate: {}".format(noise_rate))
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=1e-6)
        #self.epochs = 250


    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r, _ = self.model(inputs)
        loss = self.criterion(x_r, inputs)
        self.losses.update(loss.item(), 1)
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
                _, x_r = self.model(x)
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


class InCoteachingResnetHelper(TrainTestHelper):
    def __init__(self, n_channels, SCORE_MODE, OP_TYPE, BACKEND, mask, group=4, *args, **kwargs):
        super(InCoteachingResnetHelper, self).__init__(*args, **kwargs)
        self.method_tag = "incoteaching-resnet"

        lam = 1
        self.group = group
        print("group: ", group)
        self.n_channels = n_channels
        self.SCORE_MODE = SCORE_MODE
        self.OP_TYPE = OP_TYPE
        self.mask = mask
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
        self.num_classes = 2+3+3+4
        if BACKEND == 'wrn':
            n, k = (10, 4)
            model = WideResNet(num_classes=self.num_classes, depth=n, widen_factor=k, in_channel=n_channels)
        elif BACKEND == 'resnet20':
            n = 20
            model = ResNet(num_classes=self.num_classes, depth=n, in_channels=n_channels)
        elif BACKEND == 'resnet50':
            n = 50
            model = ResNet(num_classes=self.num_classes, depth=n, in_channels=n_channels)
        elif BACKEND == 'densenet22':
            n = 22
            model = DenseNet(num_classes=self.num_classes, depth=n, in_channels=n_channels)
        elif BACKEND == 'densenet40':
            n = 40
            model = DenseNet(num_classes=self.num_classes, depth=n, in_channels=n_channels)
        else:
            raise NotImplementedError('Unimplemented backend: {}'.format(BACKEND))
        self.print('Using backend: {} ({})'.format(type(model).__name__, BACKEND))

        self.model = model.cuda()
        self.batch_size = 128

        cudnn.benchmark = True

        # noise reduce
        self.noise_rate = 0.
        self.smooth_epoch = 0
        self.oe_scale = None

        #self.criterion = MulCEInCoTeachingLoss(noise_rate=0.1, group=(2,3,3,4), score_mode=self.SCORE_MODE).cuda()
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-6)
        self.num_workers = 8

        self.epochs = int(np.ceil(250 / transformer.n_transforms))

    def to_multask_labels(self, transformations_inds):
        tr_mul_idxs = []
        if self.OP_TYPE == 'RA':
            for is_flip, tx, ty, k_rotate in itertools.product(range(2), range(3),range(3), range(4)):
                tr_mul_idxs.append(np.array((is_flip, tx, ty, k_rotate)))
        else:
            raise NotImplementedError
        transformations_inds = list(map(lambda x: tr_mul_idxs[x], transformations_inds))
        return transformations_inds

    def generate_datasets(self, x):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x))
        return x, transformations_inds

    def transform_traindata(self, x_train):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train))
        self.trainset = transformer_dataset(train_data=x_train,
                                      train_labels=self.to_multask_labels(transformations_inds),
                                      data_transformer=self.transformer,
                                      transform=transform_train
                                    )
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        #self.criterion.ite = len(self.trainloader) // (self.transformer.n_transforms)
        self.criterion = MulCEInCoTeachingLoss(noise_rate=self.noise_rate, group=(2, 3, 3, 4), score_mode='pl_mean',
                                               iter_per_epoch=len(self.trainloader), smooth_epoch=self.smooth_epoch,
                                               oe_scale=self.oe_scale,mask_group=self.mask).cuda()
        self.print("noise_rate:{} iter_per_epoch:{} smooth_epoch:{} oe_scale:{}".format(self.noise_rate,
                                                                                        len(self.trainloader),
                                                                                        self.smooth_epoch,
                                                                                        self.oe_scale))


    def transform_testdata(self, x_test, y_test):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_test))
        self.testset = transformer_dataset(train_data=x_test,
                                     train_labels=self.to_multask_labels(transformations_inds),
                                     data_transformer=self.transformer,
                                     transform=transform_test)
        self.y_test = y_test
        self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train(self):
        self.losses = AverageMeter()
        test_in_epoch_target = self.test_in_epoch - 1
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                self.train_step(inputs, y)
                if batch_idx % self.test_in_epoch == test_in_epoch_target:
                    self.test(True)
            self.print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

    def train_step(self, x, y=None):
        y = y.view(x.size(0), -1)
        targets = []
        for i in range(y.size(1)):
            targets.append(y[:, i].cuda())
        inputs = torch.autograd.Variable(x.cuda())
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.losses.update(loss.data.cpu(), 1)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def test_step(self, x, y):
        y = y.view(x.size(0), -1)
        targets = []
        for i in range(y.size(1)):
            targets.append(y[:, i].cuda())
        inputs = torch.autograd.Variable(x.cuda())
        outputs, _ = self.model(inputs)
        scores = self.criterion.predict(outputs, targets)
        return scores

    def compute_scores(self, all_transform=True):
        res = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, y) in enumerate(self.testloader):
                inputs = torch.autograd.Variable(inputs.cuda())
                scores = self.test_step(inputs, y)
                res.append(scores)
        res = torch.cat(res, dim=0)
        preds = res.view(len(self.y_test), self.transformer.n_transforms)
        if all_transform:
            preds = preds.mean(dim=1)
        else:
            preds = preds[:, 0]
        return preds.cpu().numpy(), self.y_test


    def test(self, is_show=True, all_transform=True):
        scores, y_test = self.compute_scores(all_transform)
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

