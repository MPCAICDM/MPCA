from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from misc import AverageMeter
from eval_accuracy import simple_accuracy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import torch
from multiprocessing import Value

def softmax(input_tensor):
    act = torch.nn.Softmax(dim=1)
    return act(input_tensor).numpy()

class dataset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, test_data, test_labels, train=True,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class transformer_score_dataset(data.Dataset):
    def __init__(self, train_data, train_labels, data_transformer, aux_labels=None, transform=None,
                 target_transform=None, train_sequential=False):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data
        self.train_labels = train_labels
        self.aux_labels = aux_labels
        self.transfomer = data_transformer
        self.n_transforms =  self.transfomer.n_transforms
        self.train_sequential = train_sequential
        if train_sequential:
            self.length = self.train_data.shape[0]
            self.transform_idx = 0
            self.iter_count = Value('i', 0)
        else:
            self.length = self.train_data.shape[0] * self.transfomer.n_transforms
            assert self.length == len(self.train_labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train_sequential:
            with self.iter_count.get_lock():
                self.iter_count.value += 1
                if self.iter_count.value == self.length:
                    self.transform_idx = (self.transform_idx + 1) % self.n_transforms
                    self.iter_count.value = 0
            image_idx, transform_idx = idx, self.transform_idx
            nidx = image_idx * self.n_transforms + transform_idx
        else:
            image_idx, transform_idx = idx // self.n_transforms, idx % self.n_transforms
            nidx = idx
        img, target = self.transfomer.transform_one(self.train_data[image_idx], transform_idx).copy(), self.train_labels[nidx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.aux_labels is not None:
            return img, (target, self.aux_labels[idx])
        return img, target

class transformer_dataset(data.Dataset):
    def __init__(self, train_data, train_labels, data_transformer, aux_labels=None, transform=None,
                 target_transform=None, train_sequential=False, is_padding=False):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data
        self.train_labels = train_labels
        self.aux_labels = aux_labels
        self.transfomer = data_transformer
        self.n_transforms =  self.transfomer.n_transforms
        self.train_sequential = train_sequential
        self.is_padding = is_padding
        if train_sequential:
            self.length = self.train_data.shape[0]
            self.transform_idx = 0
            self.iter_count = Value('i', 0)
        else:
            self.length = self.train_data.shape[0] * self.transfomer.n_transforms
            assert self.length == len(self.train_labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train_sequential:
            with self.iter_count.get_lock():
                self.iter_count.value += 1
                if self.iter_count.value == self.length:
                    self.transform_idx = (self.transform_idx + 1) % self.n_transforms
                    self.iter_count.value = 0
            image_idx, transform_idx = idx, self.transform_idx
            nidx = image_idx * self.n_transforms + transform_idx
        else:
            image_idx, transform_idx = idx // self.n_transforms, idx % self.n_transforms
            nidx = idx
        if self.is_padding:
            img = np.pad(self.train_data[image_idx].copy(),  ((2, 2), (2, 2), (0, 0)), 'constant')
            #print(img.shape)
            img, target = self.transfomer.transform_one(img, transform_idx).copy(), self.train_labels[nidx]
        else:
            img, target = self.transfomer.transform_one(self.train_data[image_idx], transform_idx).copy(), self.train_labels[nidx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.aux_labels is not None:
            return img, (target, self.aux_labels[idx])
        return img, target

class h5idx_dataset(data.Dataset):
    def __init__(self, train_index, train_labels, total_data, aux_labels=None, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_index = train_index  # just a index
        self.train_labels = train_labels
        self.aux_labels = aux_labels
        self.total_data = total_data
        self.length = self.train_index.shape[0] * self.total_data.shape[1]
        self.n_transform = self.total_data.shape[1]
        assert self.length == len(self.train_labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image_idx, transform_idx = idx // self.n_transform, idx % self.n_transform
        img, target = np.array(self.total_data[self.train_index[image_idx], transform_idx, :]), self.train_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.aux_labels is not None:
            return img, (target, self.aux_labels[idx])
        return img, target


class trainset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, aux_labels=None,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels
        self.aux_labels = aux_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.aux_labels is not None:
            return img, (target, self.aux_labels[index])
        return img, target

    def __len__(self):
        return len(self.train_data)

class testset_pytorch(data.Dataset):
    def __init__(self, test_data, transform=None):
        self.transform = transform
        self.test_data = test_data  # ndarray

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.test_data)

class dataset_reorganized(data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform

        self.data = data  # ndarray

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgs = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            new_imgs = []
            for i in range(imgs.shape[0]):
                img = imgs[i]
                img = self.transform(img)
                new_imgs.append(img.unsqueeze(0))
            new_imgs = torch.cat(new_imgs, dim=0)
        else:
            raise NotImplementedError


        return new_imgs

    def __len__(self):
        return len(self.data)

def train_reorganized(trainloader, model, criterion, optimizer, epochs):
    # train the model
    model.train()
    top1 = AverageMeter()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs) in enumerate(trainloader):
            targets = torch.LongTensor(np.tile(np.arange(inputs.size(1)), inputs.size(0)))
            inputs = inputs.reshape(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))

            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())

            outputs, _ = model(inputs)

            loss = criterion(outputs, targets)

            prec1 = simple_accuracy(outputs.data.cpu(), targets.data.cpu())

            top1.update(prec1, inputs.size(0))
            losses.update(loss.data.cpu(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}, Accuracy: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg, top1.avg))

def test_reorganized(testloader, model):
    model.eval()
    res = torch.Tensor()
    for batch_idx, (inputs) in enumerate(testloader):
        inputs = inputs.reshape(-1, inputs.size(-3), inputs.size(-2), inputs.size(-1))
        inputs = torch.autograd.Variable(inputs.cuda())
        outputs, _ = model(inputs)
        res = torch.cat((res, outputs.data.cpu()), dim=0)
    return res




def get_scores(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        scores.append(outputs[i, targets[i]])
    return np.array(scores)