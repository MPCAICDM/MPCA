from helpers.cae_helper import CAEHelper
from helpers.e3outlier_helper import E3outlierHelper
from helpers.cae_lenet_helper import CAELeNetHelper
from helpers.cae_lsa_helper import CAELSAHelper
from helpers.lsa_helper import LSAHelper
from helpers.cae_gpnd_helper import CAEGPNDHelper
from helpers.e3outlier_oe_helper import E3outliterOEHelper
from helpers.coteaching_helper import CoTeachingHelper
from helpers.e3outlier_pencil_helper import E3outlierPENCILHelper
from helpers.in_coteaching_helper import InCoTeachingHelper, InCoteachingResnetHelper

import argparse
import os
from multiprocessing import Manager
import numpy as np
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis, save_false_sample
from outlier_datasets_separate import  load_mnist_with_outliers_general,load_fashion_mnist_with_outliers_general,load_cifar10_with_outliers_general
import torchvision.transforms as transforms
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch
import shutil

parser = argparse.ArgumentParser(description='Run UOD experiments.')
parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
parser.add_argument('--transform_backend', type=str, default='wrn', help='Backbone network for SSD.')
parser.add_argument('--operation_type', type=str, default='RA+IA+PR',
                    choices=['RA', 'RA+IA', 'RA+IA+PR', 'Rotate4D'], help='Type of operations.')
parser.add_argument('--score_mode', type=str, default='neg_entropy',
                    choices=['pl_mean', 'max_mean', 'neg_entropy'],
                    help='Score mode for E3Outlier: pl_mean/max_mean/neg_entropy.')
parser.add_argument('--clear_results', action='store_true', help='clear previous results')
args = parser.parse_args()
RESULTS_DIR = args.results_dir
BACKEND = args.transform_backend
OP_TYPE = args.operation_type
SCORE_MODE = args.score_mode
DATASET_DIR = "/data/ljh/datasets"

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

def _cae_gpnd_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    n_channels = x_train.shape[get_channels_axis()]
    if x_test is None:
        testset = trainset
    else:
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    cae_helper = CAEGPNDHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                                single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                                batch_size=128, test_per_epoch=1, is_save_train=True,
                                epochs=70)
    cae_helper.train()

def _lsa_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    lsa_helper = LSAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                 batch_size=128,test_per_epoch=1, is_save_train=True,
                 epochs=70, n_channels=n_channels, h=h, w=w)
    lsa_helper.train()

def _coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = CoTeachingHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=1, is_save_train=True,
                              epochs=70, n_channels=n_channels, h=h, w=w)
    cae_helper.train()

def _in_coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = InCoTeachingHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=1, is_save_train=True,
                              epochs=70, n_channels=n_channels, h=h, w=w, group=8)
    cae_helper.train()

def _in_coteaching_resnet_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction, rand_seed,
                                     dataset_mode):
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = InCoteachingResnetHelper(trainset=None, testset=None, dataset_name=dataset_name,
                                single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                                batch_size=128, test_per_epoch=1, is_save_train=True,
                                epochs=150, n_channels=n_channels, OP_TYPE=OP_TYPE, BACKEND=BACKEND,
                                SCORE_MODE=SCORE_MODE, DATASET_DIR=DATASET_DIR, rand_seed=rand_seed,
                                dataset_mode=dataset_mode)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)

def _cae_lsa_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = CAELSAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                 batch_size=128,test_per_epoch=1, is_save_train=True,
                 epochs=70, n_channels=n_channels, h=h, w=w)
    cae_helper.train()

def _cae_lenet_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    n_channels = x_train.shape[get_channels_axis()]
    if x_test is None:
        testset = trainset
    else:
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    cae_helper = CAELeNetHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                 batch_size=128,test_per_epoch=1, is_save_train=True,
                 epochs=70, n_channels=n_channels)
    cae_helper.train()
    #cae_helper.test(False)

def _cae_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    n_channels = x_train.shape[get_channels_axis()]
    if x_test is None:
        testset = trainset
    else:
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    cae_helper = CAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                 batch_size=128,test_per_epoch=1, is_save_train=True,
                 epochs=70, n_channels=n_channels)
    cae_helper.train()
    #cae_helper.test(True)
    #cae_helper.test(False)

def _e3outlier_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outlierHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=150, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test,y_test)
    e3_helper.train()
    e3_helper.test(True)

def _e3outlier_pencil_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    n_channels = x_train.shape[get_channels_axis()]
    helper = E3outlierPENCILHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=150, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE,stage1=1,stage2=10)
    helper.transform_traindata(x_train)
    helper.transform_testdata(x_test,y_test)
    helper.train()

def _e3outlier_oe_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction):
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outliterOEHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=150, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.transform_traindata(x_train, None)
    e3_helper.transform_testdata(x_test,y_test)
    e3_helper.train()
    e3_helper.test(True)


def run_general_experiments(load_dataset_fn, dataset_name, q, n_classes, data_mode, run_idx):
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    train_mode, trainp, test_mode, testp = data_mode
    dataset_mode = "".join([str(i) for i in data_mode])
    for c in list(range(n_classes)):
        np.random.seed(run_idx)
        x_train, y_train, x_test, y_test = load_dataset_fn(c, train_mode, trainp, test_mode, testp)

        # random sampling if the number of data is too large
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]

        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'lsa'))
        #_lsa_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #_cae_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'cae_lsa'))
        ##_cae_lsa_pytorch_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'cae_ori'))
        #_cae_pytorch_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'cae_lenet'))
        #_cae_lenet_pytorch_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp)
        #_cae_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, 'e3', c))
        #_e3outlier_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'cae_gpnd'))
        #_cae_gpnd_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{}".format(data_mode, dataset_name, 'e3_oe'))
        #_e3outlier_oe_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, 'coteaching', c))
        #_coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, 'e3outlier_pencil', c))
        #_e3outlier_pencil_pytorch_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp)
        #print("data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, 'in_coteaching', c))
        #_in_coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, trainp)
        print("data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, 'in_coteaching_resnet', c))
        _in_coteaching_resnet_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp,
                                         rand_seed=run_idx,dataset_mode=dataset_mode)


# Collections of all valid algorithms.
__ALGO_NAMES__ = ['{}-{}'.format(algo, p)
                  for algo in ('cae', 'cae-iforest', 'drae', 'rdae', 'dagmm', 'ssd-iforest', 'e3outlier')
                  for p in (0.05, 0.1, 0.15, 0.2, 0.25)]


if __name__ == '__main__':
    if args.clear_results:
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
        print("Clear previous results")
    n_run = 1 #5 TODO
    N_GPUS = 1  # deprecated, use one gpu only
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))

    experiments_list = [
        (load_mnist_with_outliers_general, 'mnist', 10),
        #(load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
        (load_cifar10_with_outliers_general, 'cifar10', 10),
        #(load_cifar100_with_outliers, 'cifar100', 20),
        #(load_svhn_with_outliers, 'svhn', 10),
    ]

    # p_list = [0.05, 0.1, 0.15, 0.2, 0.25] TODO
    p_list = [('SINGLE',  0, 'ALL', None),
              ('PERCENT', 0.1, 'SAME', None),
              #('PERCENT', 0.1, 'PERCENT', 0.1),
              ('PERCENT', 0.1, 'ALL', None),
              ]
    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i)