"""Dataset utilities."""
import numpy as np
import torch
import torch.utils.data
from utils import (
    load_cifar10, load_cifar100, load_mnist, load_fashion_mnist, load_svhn
)
from transformations import RA, RA_IA, RA_IA_PR, Rotate4D
import os

_dataset = {"tag":None,'y_test':None, 'y_train':None, 'data':None}

def _load_data_with_outliers(normal, abnormal, p):
    num_abnormal = int(normal.shape[0]*p/(1-p))
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1
    return data, labels

def _transform_data(x, OP_TYPE='RA'):
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
    transformer = transformer
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x))
    x_transformed = transformer.transform_batch(
        np.repeat(x, transformer.n_transforms, axis=0), transformations_inds)
    nx_shape = list(x.shape)
    nx_shape = nx_shape[:1] + [transformer.n_transforms] + nx_shape[1:]
    nx_shape = tuple(nx_shape)
    x_transformed = x_transformed.reshape(nx_shape)
    print(x_transformed.shape)
    return x_transformed

def _load_data_one_vs_all_transformed(data_load_fn, class_ind, dataset_name='',dir_path='./datasets'
                                      ,transform_mode='RA',
                                      train_mode='PERCENT', trainp=None, test_mode='SAME', testp=None):
    dataset_path = '{}_{}.npz'.format(dataset_name, transform_mode)
    dataset_path = os.path.join(dir_path, dataset_name, dataset_path)
    os.makedirs(os.path.join(dir_path, dataset_name), exist_ok=True)
    if dataset_path == _dataset['tag']:
        pass
    elif os.path.exists(dataset_path):
        dataset = np.load(dataset_path)
        _dataset['data'], _dataset['y_train'], _dataset['y_test'] = dataset['data'], dataset['y_train'],dataset['y_test']
        _dataset['tag'] = dataset_path
    else:
        (x_train, y_train), (x_test, y_test) = data_load_fn()
        x_train, x_test = _transform_data(x_train, transform_mode), _transform_data(x_test, transform_mode)
        data = np.concatenate((x_train, x_test), axis=0)
        np.savez_compressed(dataset_path, data=data, y_test=y_test, y_train=y_train)
        _dataset['data'], _dataset['y_train'], _dataset['y_test'] = data, y_train, y_test
        _dataset['tag'] = dataset_path
    y_test, y_train = _dataset['y_test'], _dataset['y_train']
    x_train = np.array(list(range(len(y_train) + len(y_test)))[:len(y_train)])
    x_test = np.array(list(range(len(y_train) + len(y_test)))[len(y_train):])
    data = _dataset['data']

    # before x_train N * C * H * W
    # after N * T * C * H * W
    if test_mode == 'SAME':
        X = np.concatenate((x_train, x_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        normal = X[Y.flatten() == class_ind]
        abnormal = X[Y.flatten() != class_ind]
        train_data, train_labels = _load_data_with_outliers(normal, abnormal, trainp)
        return train_data, train_labels,train_data, train_labels, data
    elif train_mode == 'PERCENT':
        train_normal = x_train[y_train.flatten() == class_ind]
        train_abnormal = x_train[y_train.flatten() != class_ind]
        train_data, train_labels = _load_data_with_outliers(train_normal, train_abnormal, trainp)
    elif train_mode == 'SINGLE':
        train_normal = x_train[y_train.flatten() == class_ind]
        train_data, train_labels = train_normal, np.zeros((train_normal.shape[0], ), dtype=np.int32)
    else:
        raise NotImplementedError("Error")

    if test_mode == 'PERCENT':
        test_normal = x_test[y_test.flatten() == class_ind]
        test_abnormal = x_test[y_test.flatten() != class_ind]
        test_data, test_labels = _load_data_with_outliers(test_normal, test_abnormal, testp)
    elif test_mode == 'ALL':
        test_data = x_test
        test_labels = (y_test == class_ind) + 0
    else:
        raise NotImplementedError("Error")
    return train_data, train_labels, test_data, test_labels, data


def load_mnist_with_outliers_transformed(class_ind,dir_path,transform_mode,train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None):
    return _load_data_one_vs_all_transformed(load_mnist, class_ind, "mnist",dir_path,transform_mode, train_mode, trainp, test_mode, testp)

def load_fashion_mnist_with_outliers_transformed(class_ind,dir_path,transform_mode,train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None):
    return _load_data_one_vs_all_transformed(load_fashion_mnist, class_ind, 'fashion_mnist',dir_path,transform_mode, train_mode, trainp, test_mode, testp)

def load_cifar10_with_outliers_transformed(class_ind,dir_path,transform_mode,train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None):
    return _load_data_one_vs_all_transformed(load_cifar10, class_ind, 'cifar10',dir_path,transform_mode, train_mode, trainp, test_mode, testp)



class OutlierDataset(torch.utils.data.TensorDataset):

    def __init__(self, normal, abnormal, percentage):
        """Samples abnormal data so that the total size of dataset has
        percentage of abnormal data."""
        data, labels = _load_data_with_outliers(normal, abnormal, percentage)
        super(OutlierDataset, self).__init__(
            torch.from_numpy(data), torch.from_numpy(labels)
        )
