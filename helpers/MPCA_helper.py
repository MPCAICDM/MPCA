from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data, show_avg_scores
from models.SN_mnist import SN_MNIST, RSRBoneType, \
    RSRBoneTypeV2, RSRBoneTypeV3Linear, RSRBoneTypeV3,\
    RSRBoneTypeV4Linear, RSRBoneTypeV6, RSRBoneTypeV6Linear, RSRBoneTypeV7
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
from misc import AverageMeter
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
import numpy as np
from models.wrn_pytorch import PCAWideResNet
from transformations import RA, RA_IA, RA_IA_PR, Rotate4D
from loss_functions.SN_loss import SNLossV1,MPCALossV1, MPCALossV2, MPCAGTLossV1, MPCAGTLossV2
from keras2pytorch_dataset import trainset_pytorch, transformer_dataset

class SNHelper(TrainTestHelper):
    def __init__(self, n_channels, h, w, score_norm=True, *args, **kwargs):
        super(SNHelper, self).__init__(*args, **kwargs)
        self.method_tag = "lsa"

        self.n_channels = n_channels
        lamb1 = 0.
        lamb2 = 0.
        lr = 0.00025
        mode = 'A'
        self.model = RSRBoneTypeV2(input_shape=(n_channels, h, w), z_channels=30,
                             hidden_layer_sizes=[32, 64, 128],mode=mode).cuda()

        cudnn.benchmark = True
        self.print("lam1:{} lam2:{} mode:{}".format(lamb1, lamb2, mode))
        self.criterion = MPCALossV1(lamb1=lamb1, lamb2=lamb2, A=self.model.A, B=self.model.B)
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        # self.epochs = 250

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        y, y_r, x_r = self.model(inputs)
        loss = self.criterion(inputs, x_r, y, y_r)
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        lossA = []
        y_test = []
        lossB = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                _, _, x_r = self.model.predict(x,'A')
                loss = x_r.sub(x).pow(2).view(x_r.size(0), -1)
                loss = -loss.sum(dim=1, keepdim=False)
                lossA.append(loss.data.cpu())
                y_test.append(labels.data.cpu())
                _, _, x_r = self.model.predict(x, 'B')
                loss = x_r.sub(x).pow(2).view(x_r.size(0), -1)
                loss = -loss.sum(dim=1, keepdim=False)
                lossB.append(loss.data.cpu())
        lossA = torch.cat(lossA, dim=0)
        lossB = torch.cat(lossB, dim=0)
        y_test = torch.cat(y_test, dim=0)
        scores = (lossA, lossB)
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            self.print('Atype')
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[0], y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
            self.print('Btype')
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[1], y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)


class MPCAHelper(TrainTestHelper):
    def __init__(self, n_channels, h, w,lamb1,lamb2,pca_mode,noise_rate,z_channels,
                 shareAB, train_mode, loss_mode, lr=0.00025, flatten_size=128, *args, **kwargs):
        super(MPCAHelper, self).__init__(*args, **kwargs)
        self.method_tag = "mpca"

        self.n_channels = n_channels

        mode = pca_mode
        if self.dataset_name in ['mnist', 'caltech101', 'fashion-mnist-rsrae', 'fashion-mnist',
                                 'tinyimagenet', 'cifar10']:
            # self.model = RSRBoneTypeV3(input_shape=(n_channels, h, w), z_channels=z_channels, shareAB=shareAB,
            #                                  hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()
            self.model = RSRBoneTypeV6(input_shape=(n_channels, h, w), z_channels=z_channels, shareAB=shareAB,
                                       hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False, flatten_size=flatten_size).cuda()
            #self.model = RSRBoneTypeV7(input_shape=(n_channels, h, w), z_channels=z_channels, shareAB=shareAB,
            #                           hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False, flatten_size=flatten_size).cuda()
        else:
            # self.model = RSRBoneTypeV6Linear(input_shape=n_channels, z_channels=z_channels, shareAB=shareAB,
            #                                  hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False,
            #                                  flatten_size=flatten_size).cuda()
            self.model = RSRBoneTypeV6Linear(input_shape=n_channels, z_channels=z_channels, shareAB=shareAB,
                                             hidden_layer_sizes=[512, 256, 128], mode=mode, bn=False,
                                             flatten_size=flatten_size).cuda()
            # self.model = RSRBoneTypeV3Linear(input_shape=n_channels, z_channels=z_channels, shareAB=shareAB,
            #                                  hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()
            # self.model = RSRBoneTypeV3Linear(input_shape=n_channels, z_channels=z_channels, shareAB=shareAB,
            #                                  hidden_layer_sizes=[512, 256, 128], mode=mode, bn=False).cuda()

        cudnn.benchmark = True
        self.print("lam1:{} lam2:{} mode:{} noise_rate:{}".format(lamb1, lamb2, mode, noise_rate))
        self.criterion = MPCALossV1(lamb1=lamb1, lamb2=lamb2,
                                    A=self.model.A, B=self.model.B,
                                    noise_rate=noise_rate, mode='exchange')
        self.train_mode = train_mode
        if self.train_mode == 'AD':
            self.optimizerA = optim.Adam(self.model.parameters(), weight_decay=1e-6, lr=lr)
            self.optimizerB = optim.Adam([self.model.A], weight_decay=1e-6, lr=lr)
            self.optimizerC = optim.Adam(list(self.model.encoder.parameters()) + [self.model.A], weight_decay=1e-6, lr=lr)
            # self.optimizerC = optim.Adam([{"params":self.model.encoder.parameters()}, {"params":self.model.A}],eps=1e-7,weight_decay=0.0005, lr=lr*10)
        else:
            # use adam always
            self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-6, lr=lr)
            #self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        if self.train_mode == 'AD':
            y, y_r, x_r = self.model(inputs)
            lossA = self.criterion.L21_error(inputs, x_r).mean() + \
                    self.criterion.proj_error() + \
                    self.criterion.pca_loss(y, y_r).mean()
            # print(lossA)
            self.losses.update(lossA.item(), 1)
            # compute gradient and do SGD step
            self.optimizerA.zero_grad()
            lossA.backward()
            self.optimizerA.step()

            lossB = self.criterion.proj_error()
            self.optimizerB.zero_grad()
            lossB.backward()
            self.optimizerB.step()

            y, y_r, x_r = self.model(inputs)
            lossC = self.criterion.pca_loss(y, y_r).mean()
            self.optimizerC.zero_grad()
            lossC.backward()
            self.optimizerC.step()
        elif self.train_mode == 'CAD':
            y, y_r, x_r = self.model(inputs)
            lossA = self.criterion.L21_error(inputs, x_r).mean() + \
                    self.criterion.proj_error() + \
                    self.criterion.pca_loss(y, y_r).mean()
            # print(lossA)
            self.losses.update(lossA.item(), 1)
            # compute gradient and do SGD step
            self.optimizerA.zero_grad()
            lossA.backward()
            self.optimizerA.step()

            lossB = self.criterion.proj_error()
            self.optimizerB.zero_grad()
            lossB.backward()
            self.optimizerB.step()

            y, y_r, x_r = self.model(inputs)
            lossC = self.criterion.pca_loss(y, y_r).mean()
            self.optimizerC.zero_grad()
            lossC.backward()
            self.optimizerC.step()
        else:
            y, y_r, x_r = self.model(inputs)
            loss = self.criterion(inputs, x_r, y, y_r)
            self.losses.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        scoreA = []
        y_test = []
        scoreB = []
        scoreAB = []
        scoreY = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                y, y_r, (x_rA, x_rB, x_rAB) = self.model.predict(x)
                lossA = x_rA.sub(x).pow(2).view(x_rA.size(0), -1).sum(dim=1, keepdim=False)
                lossB = x_rB.sub(x).pow(2).view(x_rB.size(0), -1).sum(dim=1, keepdim=False)
                lossAB = x_rAB.sub(x).pow(2).view(x_rAB.size(0), -1).sum(dim=1, keepdim=False)
                lossY = y.sub(y_r).pow(2).view(y.size(0), -1).sum(dim=1, keepdim=False)
                scoreA.append(-lossA.cpu())
                scoreB.append(-lossB.cpu())
                scoreAB.append(-lossAB.cpu())
                scoreY.append(-lossY.cpu())
                y_test.append(labels.cpu())

        scoreA = torch.cat(scoreA, dim=0)
        scoreB = torch.cat(scoreB, dim=0)
        scoreAB = torch.cat(scoreAB, dim=0)
        scoreY = torch.cat(scoreY, dim=0)
        y_test = torch.cat(y_test, dim=0)
        scores = (scoreA, scoreB, scoreAB, scoreY)
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            tags = ['A', 'B', 'AB', 'Y']
            for i in range(len(tags)):
                self.print('type {}'.format(tags[i]))
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

                mean_pos, mean_neg, std_pos, std_neg = show_avg_scores(scores[i].cpu().numpy(), y_test)
                self.print(
                    "mean_pos:{}, mean_neg:{} std_pos:{} std_neg:{}".format(mean_pos, mean_neg, std_pos, std_neg))
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

    def save(self, ori_tag):
        scores, y_test = self.compute_scores()
        tags = ['A', 'B', 'AB', 'Y']
        for i in range(len(tags)):
            #TODO
            res_file_path = self.get_result_file_path(tag=ori_tag+"-"+tags[i])
            print(res_file_path)
            save_roc_pr_curve_data(scores[i], y_test, res_file_path)

class MPCAV2Helper(TrainTestHelper):
    def __init__(self, n_channels, h, w,lamb1,lamb2,pca_mode,noise_rate,z_channels,
                 shareAB, *args, **kwargs):
        super(MPCAV2Helper, self).__init__(*args, **kwargs)
        self.method_tag = "mpca"

        self.n_channels = n_channels
        lr = 0.00025
        mode = pca_mode
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist']:
            self.model = RSRBoneTypeV3(input_shape=(n_channels, h, w), z_channels=z_channels, shareAB=shareAB,
                                             hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()
        else:
            self.model = RSRBoneTypeV4Linear(input_shape=n_channels, z_channels=z_channels,
                                             hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()

        cudnn.benchmark = True
        self.print("lam1:{} lam2:{} mode:{} noise_rate:{}".format(lamb1, lamb2, mode, noise_rate))
        self.criterion = MPCALossV2(lam=0.,
                                    A=self.model.inner_net.A, B=self.model.inner_net.B,
                                    noise_rate=noise_rate, mode='exchange')
        self.noise_rate = noise_rate
        # use adam always
        self.optimizerED = optim.Adam([{'params':self.model.encoder.parameters()},
                                       {'params':self.model.decoder.parameters()}],
                                      weight_decay=1e-6, lr=lr)
        self.optimizerAB = optim.Adam([{'params':self.model.encoder.parameters()},
                                      {'params':self.model.inner_net.parameters()}],
                                      weight_decay=1e-6, lr=lr*10)
        self.smooth_epoch = 30

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
        x = torch.autograd.Variable(x.cuda())

        y, y_r, x_r = self.model(x)
        lossED = self.criterion(x, x_r, y, y_r, 'ED')
        self.losses.update(lossED.item(), x.size(0))
        self.optimizerED.zero_grad()
        lossED.backward()
        self.optimizerED.step()

        y, y_r, x_r = self.model(x)
        lossAB = self.criterion(x, x_r, y, y_r, 'AB')
        self.losses.update(lossAB.item(), x.size(0))
        self.optimizerAB.zero_grad()
        lossAB.backward()
        self.optimizerAB.step()



    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        scoreA = []
        y_test = []
        scoreB = []
        scoreAB = []
        scoreY = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                y, y_r, (x_rA, x_rB, x_rAB) = self.model.predict(x)
                lossA = x_rA.sub(x).pow(2).view(x_rA.size(0), -1).sum(dim=1, keepdim=False)
                lossB = x_rB.sub(x).pow(2).view(x_rB.size(0), -1).sum(dim=1, keepdim=False)
                lossAB = x_rAB.sub(x).pow(2).view(x_rAB.size(0), -1).sum(dim=1, keepdim=False)
                lossY = y.sub(y_r).pow(2).view(y.size(0), -1).sum(dim=1, keepdim=False)
                #lossY = torch.norm(y_r-y,  p=2, dim=1)
                scoreA.append(-lossA.cpu())
                scoreB.append(-lossB.cpu())
                scoreAB.append(-lossAB.cpu())
                scoreY.append(-lossY.cpu())
                y_test.append(labels.cpu())

        scoreA = torch.cat(scoreA, dim=0)
        scoreB = torch.cat(scoreB, dim=0)
        scoreAB = torch.cat(scoreAB, dim=0)
        scoreY = torch.cat(scoreY, dim=0)
        y_test = torch.cat(y_test, dim=0)
        scores = (scoreA, scoreB, scoreAB, scoreY)
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            tags = ['A', 'B', 'AB', 'Y']
            for i in range(len(tags)):
                self.print('type {}'.format(tags[i]))
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

    def save(self, tag):
        scores, y_test = self.compute_scores()
        tags = ['A', 'B', 'AB', 'Y']
        for i in range(len(tags)):
            #TODO
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)


class MPCAGTHelper(TrainTestHelper):
    def __init__(self, n_channels,lamb1,lamb2,pca_mode,noise_rate,z_channels, proj_mode,
                 shareAB, lr, OP_TYPE='RA', SCORE_MODE='pl_mean',
                 update_mode='All', *args, **kwargs):
        super(MPCAGTHelper, self).__init__(*args, **kwargs)
        self.method_tag = "mpca-gt"

        self.n_channels = n_channels
        mode = pca_mode
        self.OP_TYPE = OP_TYPE
        self.SCORE_MODE = SCORE_MODE
        self.print(OP_TYPE)
        if self.OP_TYPE == 'RA':
            self.transformer = RA(8, 8)
        elif self.OP_TYPE == 'RA+IA+PR':
            self.transformer = RA_IA_PR(8, 8, 12, 23, 2)
        n, k = (10, 4)
        self.model = PCAWideResNet(num_classes=self.transformer.n_transforms, depth=n, widen_factor=k,
                                   in_channel=n_channels,
                                   z_channels=z_channels, mode=mode,shareAB=shareAB).cuda()
        cudnn.benchmark = True
        self.print("lam1:{} lam2:{} mode:{} noise_rate:{} shareAB:{}".format(lamb1, lamb2, mode, noise_rate, shareAB))
        self.criterion = MPCAGTLossV2(lamb1=lamb1, lamb2=lamb2,
                                    As=self.model.As, Bs=self.model.As if shareAB else self.model.Bs,
                                    noise_rate=noise_rate, mode='none', batch_size=self.batch_size, proj_mode=proj_mode)

        if update_mode == 'All':
            update_params = self.model.parameters()
            self.print("update_mode: all")
        else:
            update_params = [self.model.As]
            self.print("update_mode: A")
        # use adam always
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            #self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
            self.optimizer = optim.Adam(update_params, lr=lr, eps=1e-7, weight_decay=0.0005)
            #print('SGD')
        else:
            self.optimizer = optim.Adam(update_params, lr=lr, eps=1e-7, weight_decay=0.0005)
            #self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        # self.epochs = 250
        #self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = int(np.ceil(self.epochs / self.transformer.n_transforms))

    def transform_traindata(self, x_train):
        print('transform_traindata')
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train))
        #self.x_train_task_transformed = self.transformer.transform_batch(
        #    np.repeat(x_train, self.transformer.n_transforms, axis=0), transformations_inds)
        self.trainset = transformer_dataset(train_data=x_train,
                                         train_labels=transformations_inds,
                                            data_transformer=self.transformer,
                                            transform=transform_train, is_padding=False)
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def transform_testdata(self, x_test, y_test):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_test))
        #self.x_test_task_transformed = self.transformer.transform_batch(
        #    np.repeat(x_test, self.transformer.n_transforms, axis=0), transformations_inds)
        self.testset = transformer_dataset(train_data=x_test,
                                         train_labels=transformations_inds,
                                           data_transformer=self.transformer,
                                        transform=transform_test, is_padding=False)
        self.y_test = y_test
        #self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
        self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.num_workers)

    def train_step(self, x, y=None):
        inputs, targets = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        outputs1, y, y_rsr = self.model(inputs, targets)
        loss= self.criterion(outputs1, targets, y, y_rsr)
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_step(self, x, y):
        inputs, targets = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        (outA, outB, outAB), y, y_rsr = self.model.predict(inputs, targets)
        '''
        scoresA = self.criterion.neg_entropy(outA)
        scoresB = self.criterion.neg_entropy(outB)
        scoresAB = self.criterion.neg_entropy(outAB)
        '''
        if self.SCORE_MODE == 'pl_mean':
            scoresA = self.criterion.pl_mean(outA, targets)
            scoresB = self.criterion.pl_mean(outB, targets)
            scoresAB = self.criterion.pl_mean(outAB, targets)
        elif self.SCORE_MODE == 'neg_entropy':
            scoresA = self.criterion.neg_entropy(outA)
            scoresB = self.criterion.neg_entropy(outB)
            scoresAB = self.criterion.neg_entropy(outAB)
        #print(torch.isnan(scoresAB).sum())
        scoresY = y.sub(y_rsr).pow(2).view(y.size(0), -1).sum(dim=1, keepdim=False)

        return scoresA, scoresB, scoresAB, scoresY

    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        scoreA = []
        ##y_test = []
        scoreB = []
        scoreAB = []
        scoreY = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            with torch.no_grad():
                lossA, lossB, lossAB, lossY = self.test_step(inputs, labels)
                scoreA.append(lossA.cpu())
                scoreB.append(lossB.cpu())
                scoreAB.append(lossAB.cpu())
                scoreY.append(lossY.cpu())
                #y_test.append(labels.cpu())

        scoreA = torch.cat(scoreA, dim=0)
        scoreB = torch.cat(scoreB, dim=0)
        scoreAB = torch.cat(scoreAB, dim=0)
        scoreY = torch.cat(scoreY, dim=0)

        scoreA = scoreA.view(len(self.y_test), self.transformer.n_transforms).mean(dim=1).cpu().numpy()
        scoreB = scoreB.view(len(self.y_test), self.transformer.n_transforms).mean(dim=1).cpu().numpy()
        scoreAB = scoreAB.view(len(self.y_test), self.transformer.n_transforms).mean(dim=1).cpu().numpy()
        scoreY = scoreY.view(len(self.y_test), self.transformer.n_transforms).mean(dim=1).cpu().numpy()

        #y_test = torch.cat(y_test, dim=0)
        scores = (scoreA, scoreB, scoreAB, scoreY)
        return scores, self.y_test

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            tags = ['A', 'B', 'AB', 'Y']
            for i in range(len(tags)):
                self.print('type {}'.format(tags[i]))
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

                mean_pos, mean_neg, std_pos, std_neg = show_avg_scores(scores[i], y_test)
                self.print(
                    "mean_pos:{}, mean_neg:{} std_pos:{} std_neg:{}".format(mean_pos, mean_neg, std_pos, std_neg))

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

    def save(self, ori_tag):
        scores, y_test = self.compute_scores()
        tags = ['A', 'B', 'AB', 'Y']
        for i in range(len(tags)):
            #TODO
            res_file_path = self.get_result_file_path(tag=ori_tag+"-"+tags[i])
            print(res_file_path)
            save_roc_pr_curve_data(scores[i], y_test, res_file_path)


class MPCAV3Helper(TrainTestHelper):
    def __init__(self, n_channels, h, w,lamb1,lamb2,pca_mode,noise_rate,z_channels,
                 shareAB, *args, **kwargs):
        super(MPCAV3Helper, self).__init__(*args, **kwargs)
        self.method_tag = "mpcav3"

        self.n_channels = n_channels
        lr = 0.00025
        mode = pca_mode
        if self.dataset_name in ['mnist', 'caltech101', 'fashion-mnist-rsrae', 'fashion-mnist',
                                 'tinyimagenet', 'cifar10']:
            self.model = RSRBoneTypeV3(input_shape=(n_channels, h, w), z_channels=z_channels, shareAB=shareAB,
                                             hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()
        else:
            self.model = RSRBoneTypeV3Linear(input_shape=n_channels, z_channels=z_channels, shareAB=shareAB,
                                             hidden_layer_sizes=[32, 64, 128], mode=mode, bn=False).cuda()

        cudnn.benchmark = True
        self.print("lam1:{} lam2:{} mode:{} noise_rate:{}".format(lamb1, lamb2, mode, noise_rate))
        self.criterion = MPCALossV1(lamb1=lamb1, lamb2=lamb2,
                                    A=self.model.A, B=self.model.B,
                                    noise_rate=noise_rate, mode='exchange')
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-6, lr=lr)
        # self.epochs = 250

    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        y, y_r, x_r = self.model(inputs)
        loss = self.criterion(inputs, x_r, y, y_r)
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        # reloss = np.zeros(shape=len(self.testloader.dataset, ))
        scoreA = []
        y_test = []
        scoreB = []
        scoreAB = []
        scoreY = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                y, y_r, (x_rA, x_rB, x_rAB) = self.model.predict(x)
                lossA = x_rA.sub(x).pow(2).view(x_rA.size(0), -1).sum(dim=1, keepdim=False)
                lossB = x_rB.sub(x).pow(2).view(x_rB.size(0), -1).sum(dim=1, keepdim=False)
                lossAB = x_rAB.sub(x).pow(2).view(x_rAB.size(0), -1).sum(dim=1, keepdim=False)
                lossY = y.sub(y_r).pow(2).view(y.size(0), -1).sum(dim=1, keepdim=False)
                scoreA.append(-lossA.cpu())
                scoreB.append(-lossB.cpu())
                scoreAB.append(-lossAB.cpu())
                scoreY.append(-lossY.cpu())
                y_test.append(labels.cpu())

        scoreA = torch.cat(scoreA, dim=0)
        scoreB = torch.cat(scoreB, dim=0)
        scoreAB = torch.cat(scoreAB, dim=0)
        scoreY = torch.cat(scoreY, dim=0)
        y_test = torch.cat(y_test, dim=0)
        scores = (scoreA, scoreB, scoreAB, scoreY)
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            tags = ['A', 'B', 'AB', 'Y']
            for i in range(len(tags)):
                self.print('type {}'.format(tags[i]))
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)


