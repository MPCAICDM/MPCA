import os
from datetime import datetime
from utils import get_class_name_from_index

import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from misc import AverageMeter

from abc import abstractmethod, ABCMeta

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])


class TrainTestHelper(object):
    __metaclass__ = ABCMeta

    def __init__(self, trainset, testset, dataset_name, single_class_ind, p, RESULTS_DIR,
                 batch_size,
                 epochs=100, test_per_epoch=1, is_save_train=True, test_in_epoch=9999999,
                 num_works=8, dataset_mode="None",run_id=0, *args, **kwargs):
        #self.x_train = x_train
        #self.y_train = y_train
        self.trainset = trainset
        self.testset = testset
        self.dataset_name = dataset_name
        self.single_class_ind = single_class_ind
        self.p = p
        self.losses = None
        self.model = None
        self.trainloader = None
        self.optimizer = None
        self.criterion = None
        self.test_per_epoch = test_per_epoch
        self.is_save_train = is_save_train
        self.method_tag = "None"
        self.RESULTS_DIR = RESULTS_DIR
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_in_epoch = test_in_epoch
        self.ret_logger = []
        self.dataset_mode = dataset_mode
        self.run_id = run_id
        self.num_workers = num_works
        if self.trainset is not None:
            self.trainloader = data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_works)
        if self.testset is not None:
            self.testloader = data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=num_works)

    @abstractmethod
    def train_step(self, x, y=None):
        raise NotImplementedError("train_step not implement")

    @abstractmethod
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

            '''
            res_file_name = '{}_{}-{}_{}_{}.npz'.format(self.dataset_name, self.method_tag, self.p,
                                                            get_class_name_from_index(self.single_class_ind, self.dataset_name),
                                                            datetime.now().strftime('%Y-%m-%d-%H%M'))
            res_file_path = os.path.join(self.RESULTS_DIR, self.dataset_name, res_file_name)
            os.makedirs(os.path.join(self.RESULTS_DIR, self.dataset_name), exist_ok=True)
            save_roc_pr_curve_data(scores, self.y_train, res_file_path)
            '''

    @abstractmethod
    def test(self, is_show=True):
        raise NotImplementedError("train_step not implement")

    @abstractmethod
    def load_weight(self, pat=None, tag="None"):
        if pat is None:
            pat = '{}_{}_{}-r{}-{}_{}_{}.pth'.format(self.dataset_name, self.dataset_mode, self.method_tag, self.run_id,
                                                  self.p,
                                                  get_class_name_from_index(self.single_class_ind, self.dataset_name),
                                                     tag)
            pat = os.path.join(self.RESULTS_DIR, self.dataset_name, pat)
        self.model.load_state_dict(torch.load(pat))

    @abstractmethod
    def save_weight(self, pat=None, tag="None"):
        if pat is None:
            pat = '{}_{}_{}-r{}-{}_{}_{}.pth'.format(self.dataset_name,
                                                     self.dataset_mode, self.method_tag, self.run_id, self.p,
                                                            get_class_name_from_index(self.single_class_ind, self.dataset_name)
                                                     ,tag)
            pat = os.path.join(self.RESULTS_DIR, self.dataset_name, pat)
            os.makedirs(os.path.join(self.RESULTS_DIR, self.dataset_name), exist_ok=True)
        torch.save(self.model.state_dict(), pat)

    @abstractmethod
    def get_result_file_path(self, tag=None):
        if tag is not None:
            tag = self.method_tag + "-" + tag
        else:
            tag = self.method_tag
        res_file_name = '{}_{}_{}_{}_{}.npz'.format(self.dataset_name, tag,self.p,
                                                     get_class_name_from_index(self.single_class_ind, self.dataset_name),
                                                    self.run_id)
        #数据集_方法名+自定义tag_异常比例_异常类别_第几次重复实验
        res_file_path = os.path.join(self.RESULTS_DIR, self.dataset_name, res_file_name)
        os.makedirs(os.path.join(self.RESULTS_DIR, self.dataset_name), exist_ok=True)
        return res_file_path

    @abstractmethod
    def print(self, txt, is_show=True):
        if is_show:
            print(txt)
        self.ret_logger.append(str(txt))