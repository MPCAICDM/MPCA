from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.encoders_decoders import CAE_pytorch
from keras2pytorch_dataset import trainset_pytorch
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
from loss_functions.e3outlier_oe_loss import E3outlierOELoss
from misc import AverageMeter

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score)

class E3outlierPENCILHelper(TrainTestHelper):
    def __init__(self, n_channels,OP_TYPE,BACKEND,SCORE_MODE,stage1,stage2, *args, **kwargs):
        super(E3outlierPENCILHelper, self).__init__(*args, **kwargs)
        self.method_tag = "e3outlier_pencil"

        self.stage1 = 8
        self.stage2 = 100
        self.alpha = 0.4
        self.beta = 0.1
        self.lambda1 = 1
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
        print(transformer.n_transforms)
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
        self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss().cuda()

        # use adam always
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        else:
            self.optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
        self.epochs = int(np.ceil(250 / transformer.n_transforms))
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def transform_traindata(self, x_train):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_train))
        x_index = np.arange(len(transformations_inds))
        self.x_train_task_transformed = self.transformer.transform_batch(
            np.repeat(x_train, self.transformer.n_transforms, axis=0), transformations_inds)
        self.trainset = trainset_pytorch(train_data=self.x_train_task_transformed,
                                         train_labels=transformations_inds,aux_labels=x_index, transform=transform_train)
        self.is_y_train = True
        self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.y_float = torch.zeros(len(transformations_inds), self.transformer.n_transforms)\
            .scatter_(1, torch.from_numpy(transformations_inds).view(-1, 1), 10.0).cuda()



    def transform_testdata(self, x_test, y_test):
        transformations_inds = np.tile(np.arange(self.transformer.n_transforms), len(x_test))
        self.x_test_task_transformed = self.transformer.transform_batch(
            np.repeat(x_test, self.transformer.n_transforms, axis=0), transformations_inds)
        self.testset = trainset_pytorch(train_data=self.x_test_task_transformed,
                                         train_labels=transformations_inds, transform=transform_test)
        self.y_test = y_test
        #self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.losses = AverageMeter()
        for epoch in range(self.epochs):
            if epoch % self.test_per_epoch == 0:
                self.test(True)
            self.model.train()
            for batch_idx, (inputs, y) in enumerate(self.trainloader):
                inputs, targets, indexs = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(y[0].cuda()), \
                                          torch.autograd.Variable(y[1].cuda())
                outputs, _ = self.model(inputs)
                if epoch < self.stage1:
                    loss = self.criterion(outputs, targets)
                else:
                    yy = self.y_float[indexs, :]
                    yy = torch.autograd.Variable(yy, requires_grad=True)
                    y_hat = self.softmax(yy)
                    loss = torch.mean(self.softmax(outputs) * (self.logsoftmax(outputs) - torch.log((y_hat)))) # Loss_c
                    if epoch < self.stage2:
                        lo = self.criterion(y_hat, targets)
                        le = - torch.mean(torch.mul(self.softmax(outputs), self.logsoftmax(outputs)))
                        loss = loss + self.alpha * lo + self.beta * le

                self.losses.update(loss.data.cpu(), inputs.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch >= self.stage1 and epoch < self.stage2:
                    # update y_tilde by back-propagation
                    #yy.data.sub_(self.lambda1 * yy.grad.data)
                    self.y_float[indexs, :].sub_(self.lambda1 * yy.grad)
            print('Epoch: [{} | {}], loss: {}'.format(epoch + 1, self.epochs,
                                                                         self.losses.avg))

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
                preds[:, t] = original_preds[t, :, :][:, t]
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
            show_roc_pr_curve_data(scores, y_test)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)



