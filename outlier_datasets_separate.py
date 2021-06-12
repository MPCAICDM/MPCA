"""Dataset utilities."""
import numpy as np
import torch
import torch.utils.data
from utils import load_cifar10, load_cifar100, load_mnist, load_fashion_mnist, load_svhn, load_fashion_mnist_nopad,\
load_caltech101, load_20news,load_reuters,load_tinyimagenet
import pickle
from sklearn.preprocessing import normalize as nmlz


def _load_data_with_outliers(normal, abnormal, p, is_nmlz, is_shuffle=False):
    num_abnormal = int(normal.shape[0]*p/(1-p))
    if is_shuffle:
        selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)
        data = np.concatenate((normal, abnormal[selected]), axis=0)
        print('after shuffle:{}'.format(np.random.randint(12345678)))
    else:
        data = np.concatenate((normal, abnormal[:num_abnormal]), axis=0) #TODO
    if is_nmlz:
        (b, h, w, c) = data.shape
        data = np.reshape(data, (data.shape[0], -1))
        data = nmlz(data)
        data = np.reshape(data, (-1, h, w, c))
        print('nmlz')
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1
    return data, labels


def _load_data_one_vs_all(data_load_fn, class_ind, p, is_nmlz, is_shuffle):
    (x_train, y_train), (x_test, y_test) = data_load_fn()
    #X = np.concatenate((x_train, x_test), axis=0)
    #Y = np.concatenate((y_train, y_test), axis=0)
    train_normal = x_train[y_train.flatten() == class_ind]
    train_abnormal = x_train[y_train.flatten() != class_ind]
    train_data, train_labels =  _load_data_with_outliers(train_normal, train_abnormal, p, is_nmlz, is_shuffle)
    #test_normal = x_test[y_test.flatten() == class_ind]
    #test_abnormal = x_test[y_test.flatten() != class_ind]
    #test_data, test_labels = _load_data_with_outliers(test_normal, test_abnormal, 0.3)
    test_data, test_labels = x_test, y_test

    return train_data, train_labels, test_data, test_labels

def _load_data_one_vs_all_general(data_load_fn, class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, is_nmlz=False, is_shuffle=True):
    (x_train, y_train), (x_test, y_test) = data_load_fn(padding)
    if test_mode == 'SAME':
        if train_mode=='TEST':
            X = x_test
            Y = y_test
            trainp = trainp / (1. + trainp)
        else:
            X = np.concatenate((x_train, x_test), axis=0)
            Y = np.concatenate((y_train, y_test), axis=0)
        #X = x_test
        #Y = y_test
        normal = X[Y.flatten() == class_ind]
        abnormal = X[Y.flatten() != class_ind]
        train_data, train_labels = _load_data_with_outliers(normal, abnormal, trainp, is_nmlz, is_shuffle)
        return train_data, train_labels,train_data, train_labels
    elif train_mode == 'PERCENT':
        train_normal = x_train[y_train.flatten() == class_ind]
        train_abnormal = x_train[y_train.flatten() != class_ind]
        train_data, train_labels = _load_data_with_outliers(train_normal, train_abnormal, trainp, is_nmlz, is_shuffle)
    elif train_mode == 'SINGLE':
        train_normal = x_train[y_train.flatten() == class_ind]
        train_data, train_labels = train_normal, np.zeros((train_normal.shape[0], ), dtype=np.int32)
    else:
        raise NotImplementedError("Error")

    if test_mode == 'PERCENT':
        test_normal = x_test[y_test.flatten() == class_ind]
        test_abnormal = x_test[y_test.flatten() != class_ind]
        test_data, test_labels = _load_data_with_outliers(test_normal, test_abnormal, testp, is_nmlz, is_shuffle)
    elif test_mode == 'ALL':
        test_data = x_test
        test_labels = (y_test == class_ind) + 0
    else:
        raise NotImplementedError("Error")
    return train_data, train_labels, test_data, test_labels


def load_mnist_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_mnist, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_fashion_mnist_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_fashion_mnist, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_cifar10_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_cifar10, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_cifar100_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_cifar100, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_caltech101_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_caltech101, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_20news_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_20news, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_tinyimagenet_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_tinyimagenet, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)

def load_reuters_with_outliers_general(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    return _load_data_one_vs_all_general(load_reuters, class_ind, train_mode, trainp, test_mode, testp, padding, nmlz)


def load_fashion_mnist_with_outliers_rsrae(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    #print(class_ind, trainp)
    _, (X_test_origin, y_test_origin) = load_fashion_mnist_nopad()
    y_test = (np.array(y_test_origin) == class_ind).astype(int)
    X_test_normal = X_test_origin[y_test == 1]
    num_anomaly = int( 1000 * trainp )
    X_test_anomaly = X_test_origin[y_test == 0][0:num_anomaly]
    X_test = np.concatenate((X_test_normal, X_test_anomaly))

    labels = np.zeros((len(X_test),), dtype=np.int32)
    labels[:len(X_test_normal)] = 1
    y_test = labels

    return X_test, y_test, X_test, y_test

def load_caltech101_with_outliers_rsrae(class_ind, train_mode='PERCENT', trainp=None,
                                  test_mode='SAME', testp=None, padding=True, nmlz=False):
    with open("data/caltech101.data", 'rb') as f:
        data = pickle.load(f)
    X_test_origin = data["X"]
    y_test_origin = data["y"]
    y_test = (np.array(y_test_origin) == class_ind).astype(int)
    X_test_normal = X_test_origin[y_test == 1]
    num_anomaly = int( 1000 * trainp )
    X_test_anomaly = X_test_origin[y_test == 0][0:num_anomaly]
    X_test = np.concatenate((X_test_normal, X_test_anomaly))

    labels = np.zeros((len(X_test),), dtype=np.int32)
    labels[:len(X_test_normal)] = 1
    y_test = labels

    return X_test, y_test, X_test, y_test

class OutlierDataset(torch.utils.data.TensorDataset):

    def __init__(self, normal, abnormal, percentage):
        """Samples abnormal data so that the total size of dataset has
        percentage of abnormal data."""
        data, labels = _load_data_with_outliers(normal, abnormal, percentage)
        super(OutlierDataset, self).__init__(
            torch.from_numpy(data), torch.from_numpy(labels)
        )


def load_cifar10_with_outliers_seperate(class_ind, p):
    return _load_data_one_vs_all(load_cifar10, class_ind, p)


def load_cifar100_with_outliers_seperate(class_ind, p):
    return _load_data_one_vs_all(load_cifar100, class_ind, p)


def load_mnist_with_outliers_seperate(class_ind, p):
    return _load_data_one_vs_all(load_mnist, class_ind, p)


def load_fashion_mnist_with_outliers_seperate(class_ind, p):
    return _load_data_one_vs_all(load_fashion_mnist, class_ind, p)

def load_svhn_with_outliers_seperate(class_ind, p):
    return _load_data_one_vs_all(load_svhn, class_ind, p)
