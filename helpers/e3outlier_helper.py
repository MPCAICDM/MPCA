from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data,show_avg_scores
from models.encoders_decoders import CAE_pytorch
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
import numpy as np
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score+1e-12)

class E3outlierV2Helper(TrainTestHelper):
    def __init__(self, n_channels,OP_TYPE,BACKEND,SCORE_MODE, *args, **kwargs):
        super(E3outlierV2Helper, self).__init__(*args, **kwargs)
        self.method_tag = "e3outlier_v2"

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
            self.model = WideResNet(num_classes=transformer.n_transforms, depth=n, widen_factor=k, in_channel=n_channels).cuda()
        else:
            raise NotImplementedError('Unimplemented backend: {}'.format(BACKEND))
        print('Using backend: {} ({})'.format(type(self.model).__name__, BACKEND))

        self.num_workers = 8
        self.batch_size = 128
        self.softmax = nn.Softmax(dim=1)

        cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        # use adam always
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
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

    def train_step(self, x, y=None):
        inputs, targets = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        outputs, _ = self.model(inputs)
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.losses.update(loss.data.cpu(), inputs.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def neg_entropy(self, x):
        ne = self.softmax(x)
        return (ne *torch.log2(ne)).sum(dim=1)

    def test_step(self, x, y):
        y = y.view(x.size(0), -1)
        targets = []
        for i in range(y.size(1)):
            targets.append(y[:, i].cuda())
        inputs = torch.autograd.Variable(x.cuda())
        outputs, _ = self.model(inputs)
        scores = self.neg_entropy(outputs)
        return scores

    def compute_scores(self):
        res = []
        self.model.eval()
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

class E3outlierHelper(TrainTestHelper):
    def __init__(self, n_channels,OP_TYPE,BACKEND,SCORE_MODE, *args, **kwargs):
        super(E3outlierHelper, self).__init__(*args, **kwargs)
        self.method_tag = "e3outlier"

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
        self.print(OP_TYPE)
        self.print(transformer.n_transforms)
        self.print(SCORE_MODE)
        self.transformer = transformer

        self.BACKEND = BACKEND
        if BACKEND == 'wrn':
            n, k = (10, 4)
            model = WideResNet(num_classes=transformer.n_transforms, depth=n, widen_factor=k, in_channel=n_channels)
        elif BACKEND == 'resnet20':
            n = 20
            model = ResNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
        elif BACKEND == 'resnet50':
            n = 50
            model = ResNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
        elif BACKEND == 'densenet22':
            n = 22
            model = DenseNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
        elif BACKEND == 'densenet40':
            n = 40
            model = DenseNet(num_classes=transformer.n_transforms, depth=n, in_channels=n_channels)
        else:
            raise NotImplementedError('Unimplemented backend: {}'.format(BACKEND))
        print('Using backend: {} ({})'.format(type(model).__name__, BACKEND))

        self.model = model.cuda()
        #self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        # use adam always
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            #self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
            self.optimizer = optim.Adam(model.parameters(), eps=1e-7, lr=0.0002, weight_decay=0.0005)
        else:
            self.optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = int(np.ceil(self.epochs / transformer.n_transforms))
        self.print('epochs: {}'.format(self.epochs))

    def transform_traindata(self, x_train):
        print('transform_traindata')
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train))
        self.x_train_task_transformed = self.transformer.transform_batch(
            np.repeat(x_train, self.transformer.n_transforms, axis=0), transformations_inds)
        self.trainset = trainset_pytorch(train_data=self.x_train_task_transformed,
                                         train_labels=transformations_inds, transform=transform_train)
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        #self.test_in_epoch = len(self.trainloader) * 6 // (self.transformer.n_transforms)
        #print(self.test_in_epoch)

    def transform_testdata(self, x_test, y_test):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_test))
        self.x_test_task_transformed = self.transformer.transform_batch(
            np.repeat(x_test, self.transformer.n_transforms, axis=0), transformations_inds)
        self.testset = trainset_pytorch(train_data=self.x_test_task_transformed,
                                         train_labels=transformations_inds, transform=transform_test)
        self.y_test = y_test
        #self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)


    def train_step(self, x, y=None):
        inputs, targets = torch.autograd.Variable(x.cuda()), torch.autograd.Variable(y.cuda())
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.losses.update(loss.data.cpu(), inputs.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_features_pytorch(self):
        self.model.eval()
        features = []
        y_test = []
        for inputs,labels in self.testloader:
            inputs = torch.autograd.Variable(inputs.cuda())
            _, rep = self.model(inputs)
            features.append(rep.data.cpu())
            y_test.append(labels.data.cpu())
        features = torch.cat(features, dim=0)
        y_test = torch.cat(y_test, dim=0)
        return features.numpy(), y_test.numpy()

    def compute_softmax(self, testloader):
        self.model.eval()
        res = torch.Tensor()
        for batch_idx, (inputs) in enumerate(testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            outputs, _ = self.model(inputs)
            res = torch.cat((res, outputs.data.cpu()), dim=0)
        return nn.Softmax(dim=1)(res).numpy()

    def compute_scores(self):
        # E3Outlier
        preds = np.zeros((len(self.y_test), self.transformer.n_transforms))
        original_preds = np.zeros((self.transformer.n_transforms, len(self.y_test), self.transformer.n_transforms))
        for t in range(self.transformer.n_transforms):
            idx = np.squeeze(np.array([range(len(self.y_test))]) * self.transformer.n_transforms + t)
            test_set = testset_pytorch(test_data=self.x_test_task_transformed[idx, :],
                                       transform=transform_test)

            original_preds[t, :, :] = self.compute_softmax(data.DataLoader(test_set,
                                                                           batch_size=self.batch_size, shuffle=False))
            if self.SCORE_MODE == 'pl_mean':
                preds[:, t] = np.log(original_preds[t, :, :][:, t])
            elif self.SCORE_MODE == 'max_mean':
                preds[:, t] = np.max(original_preds[t, :, :], axis=1)
            elif self.SCORE_MODE == 'neg_entropy':
                for s in range(len(self.y_test)):
                    preds[s, t] = neg_entropy(original_preds[t, s, :])
            else:
                raise NotImplementedError
        scores = preds.mean(axis=-1)
        return scores, self.y_test

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

            mean_pos, mean_neg, std_pos, std_neg = show_avg_scores(scores, y_test)
            self.print(
                "mean_pos:{}, mean_neg:{} std_pos:{} std_neg:{}".format(mean_pos, mean_neg, std_pos, std_neg))

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

            # SSD-IF
            x_train_task_rep, y_test = self.get_features_pytorch()

            clf = IsolationForest(contamination=self.p, n_jobs=4).fit(x_train_task_rep)
            if_scores = clf.decision_function(x_train_task_rep)
            iforest_file_path = self.get_result_file_path('iforest')
            save_roc_pr_curve_data(if_scores, y_test, iforest_file_path)

    def save(self):
        scores, y_test = self.compute_scores()
        res_file_path = self.get_result_file_path()
        print(res_file_path)
        save_roc_pr_curve_data(scores, y_test, res_file_path)