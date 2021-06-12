from helpers.coteaching_helper import CoTeachingHelper, CoTeachingResnetHelper
from helpers.e3outlier_pencil_helper import E3outlierPENCILHelper
from helpers.in_coteaching_helper import InCoTeachingHelper, InCoteachingResnetHelper, InCoTeachingHiddenHelper
from helpers.estrsrae_helper import EstRSRAEHelper
from helpers.cae_lsa_helper import CAELSAHelper
from helpers.cae_helper import CAEHelper
from helpers.e3outlier_helper import E3outlierHelper, E3outlierV2Helper
from helpers.rsrae_helper import RSRAEHelper
from helpers.lsa_helper import LSAHelper
from helpers.MPCA_helper import SNHelper, MPCAHelper, MPCAV2Helper, MPCAGTHelper
from helpers.MTQ_helper import MTQHelper
from helpers.drae_helper import DRAEHelper
from helpers.dagmm_helper import DAGMMHelper

from datetime import datetime
import argparse
import os
import itertools
from multiprocessing import Manager, freeze_support, Process
import numpy as np
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis, save_false_sample
from outlier_datasets_separate import load_mnist_with_outliers_general, load_cifar10_with_outliers_general, \
    load_cifar100_with_outliers_general, load_fashion_mnist_with_outliers_general,load_fashion_mnist_with_outliers_rsrae, \
    load_caltech101_with_outliers_general, load_20news_with_outliers_general, load_reuters_with_outliers_general, \
    load_tinyimagenet_with_outliers_general
import torchvision.transforms as transforms
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch
import shutil

parser = argparse.ArgumentParser(description='Run UOD experiments.')
parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')
parser.add_argument('--dataset_dir', type=str, default='./datasets', help='Directory to save results.')
parser.add_argument('--transform_backend', type=str, default='wrn', help='Backbone network for SSD.')
parser.add_argument('--operation_type', type=str, default='RA+IA+PR',
                    choices=['RA', 'RA+IA', 'RA+IA+PR', 'Rotate4D'], help='Type of operations.')
parser.add_argument('--score_mode', type=str, default='neg_entropy',
                    choices=['pl_mean', 'max_mean', 'neg_entropy'],
                    help='Score mode for E3Outlier: pl_mean/max_mean/neg_entropy.')
parser.add_argument('--clear_results', action='store_true', help='clear previous results')
args = parser.parse_args()
#RESULTS_DIR = args.results_dir
RESULTS_DIR = '/data/linjinghuang/MPCA_results/log_scores'
BACKEND = args.transform_backend
OP_TYPE = args.operation_type
SCORE_MODE = args.score_mode
DATASET_DIR = args.dataset_dir

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

_Noise_rate = 0.1
_Group = 2
_Zsize = 20


def _cae_baseline_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = RSRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=10, is_save_train=True,
                           epochs=200,input_shape=(h, w, n_channels), z_channels=20,
                             hidden_layer_sizes=[32, 64, 128])
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _dagmm_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = x_train#trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = x_test#trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        #trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        #testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = DAGMMHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=100, is_save_train=False,
                              epochs=1000, n_channels=n_channels, h=h, w=w, y_test=y_test)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _drae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = DRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=100, is_save_train=False,
                              epochs=1000, n_channels=n_channels, h=h, w=w)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _estrsrae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = EstRSRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=10, is_save_train=False,lamb1=_Lam,mode=_Mode,
                              epochs=1000, n_channels=n_channels,noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)))
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _MPCA_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = MPCAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=100, is_save_train=True,h=h,w=w,
                           epochs=1000,n_channels=n_channels, z_channels=_Z_size,num_works=1,
                             hidden_layer_sizes=[32, 64, 128], lamb1=_Lamb1, lamb2=_Lamb2,
                             noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)),
                            pca_mode=_Pca_mode, shareAB=_ShareAB, train_mode=_TrainMode,
                            loss_mode=_LossMode, lr=_Lr, flatten_size=_Flatten_size)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    cae_helper.save(ori_tag=_Pca_mode)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _MPCA_GT_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    e3_helper = MPCAGTHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=10,n_channels=n_channels, z_channels=_Z_size,num_works=8,
                             hidden_layer_sizes=[32, 64, 128], lamb1=_Lamb1, lamb2=_Lamb2,
                             noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)),
                            pca_mode=_Pca_mode, shareAB=_ShareAB, proj_mode=_Proj_mode, lr=_Lr,
                             SCORE_MODE=_Score_Mode, OP_TYPE=_OP_TYPE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
    e3_helper.save(ori_tag=_Pca_mode)
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _MPCAV2_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = MPCAV2Helper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=10, is_save_train=True,h=h,w=w,
                           epochs=1000,n_channels=n_channels, z_channels=_Z_size,num_works=1,
                             hidden_layer_sizes=[32, 64, 128], lamb1=None, lamb2=_Lamb2,
                             noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)),
                            pca_mode=_Pca_mode, shareAB=_ShareAB)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _rsrae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
   # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    cae_helper = RSRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=10, is_save_train=True,
                           epochs=1000,input_shape=(h, w, n_channels), z_channels=10,num_works=1,
                             hidden_layer_sizes=[32, 64, 128], lamb1=_Lamb1, lamb2=_Lamb2,
                             noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)), op_type=_Op_Type)
    cae_helper.print(ret_tag)
    cae_helper.print("advance")
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _cae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    n_channels = x_train.shape[get_channels_axis()]

    cae_helper = CAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=200, n_channels=n_channels)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _lsa_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                              gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    print(x_train.shape)
    print(x_test.shape)
    #x_train = (x_train + 1.)/ 2.
    #x_test = (x_test + 1.) / 2.
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = LSAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=256, test_per_epoch=10, is_save_train=True,
                              epochs=200, n_channels=n_channels, h=h, w=w, score_norm=True)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _MTQ_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                              gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    print(x_train.shape)
    print(x_test.shape)
    #x_train = (x_train + 1.)/ 2.
    #x_test = (x_test + 1.) / 2.
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = MTQHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=10, is_save_train=True,
                              epochs=200, n_channels=n_channels, h=h, w=w, score_norm=True)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _cae_lsa_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = CAELSAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=10, is_save_train=True,
                              epochs=200, n_channels=n_channels, h=h, w=w)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _incoteaching_hidden_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    print(x_train.shape)
    if len(x_train.shape) <= 2:
        n_channels, h, w = x_train.shape[1], None, None
        trainset = trainset_pytorch(train_data=x_train, train_labels=y_train)
        testset = trainset_pytorch(train_data=x_test, train_labels=y_test)
    elif get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]

    cae_helper = InCoTeachingHiddenHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=10, is_save_train=False,group=1,lamb1=_Lam,
                                add_conv=True,
                              epochs=200, n_channels=n_channels, h=h, w=w,noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)))
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _SN_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    print(x_train.shape)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = SNHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=10, is_save_train=False,
                              epochs=200, n_channels=n_channels, h=h, w=w)
    cae_helper.print(ret_tag)
    cae_helper.print("rsrae_backbone")
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _coteaching_resnet_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                  gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = CoTeachingResnetHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=150, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _e3outlier_v2_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                  gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outlierV2Helper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=150, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _e3outlier_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                  gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outlierHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=1000, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
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
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _in_coteaching_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                              gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    noise_rate = 0.1
    cae_helper = InCoTeachingHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=1, is_save_train=True,dataset_mode=dataset_mode,
                              epochs=70, n_channels=n_channels, h=h, w=w, group=_Group, noise_rate=noise_rate)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    cae_helper.save_weight(tag='coteachnr{}g{}'.format(noise_rate,_Group))
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _in_coteaching_resnet_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                    gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    print(x_train.shape)
    print(x_test.shape)
    e3_helper = InCoteachingResnetHelper(trainset=None, testset=None, dataset_name=dataset_name,
                                single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                                batch_size=128, test_per_epoch=1, is_save_train=True,dataset_mode=dataset_mode,
                                epochs=150, n_channels=n_channels, OP_TYPE=OP_TYPE, BACKEND=BACKEND,
                                SCORE_MODE=SCORE_MODE, mask=[True, False, False, False])
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.load_weight()
    #e3_helper.train()
    for score_mode, all_tr in itertools.product(('pl_mean', 'neg_entropy'), (True, False)):
        e3_helper.print('score_mode: {} all_transform:{}'.format(score_mode, all_tr))
        e3_helper.criterion.score_mode = score_mode
        e3_helper.test(True, all_tr)
    #e3_helper.save_weight()
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)


def run_general_experiments(load_dataset_fn, dataset_name, q, n_classes, data_mode, run_idx, experiments_funcs, param_tag):
    ret_dict = man.dict()
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    train_mode, trainp, test_mode, testp = data_mode
    dataset_mode = "".join([str(i) for i in data_mode])
    print(dataset_mode)
    processes = []

    #experiments_funcs =
    for c in list(range(n_classes)):
        np.random.seed(run_idx) # run_idx
        x_train, y_train, x_test, y_test = load_dataset_fn(c, train_mode, trainp, test_mode, testp,
                                                           padding=True, nmlz=False)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        for func_iter in experiments_funcs:
            gpu_to_use = q.get()
            # random sampling if the number of data is too large
            if x_train.shape[0] > max_sample_num:
                selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
                x_train = x_train[selected, :]
                y_train = y_train[selected]
            #length = int((x_train.shape[0] - 1000)*0.2)
            #x_train = x_train[:360]
            #y_train = y_train[:360]
            #print(x_train.max(), x_train.min())
            #_in_coteaching_resnet_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp,
            #                                 run_idx,dataset_mode, gpu_to_use, q)
            def exp_func(efunc):
                ret_tag = "data_mode:{} dataset:{} exp:{} tag:{} run:{} c:{}".format(data_mode, dataset_name,
                                                                              efunc.__name__, param_tag, run_idx, c)
                process = Process(target=efunc, args=(x_train, y_train, x_test, y_test, dataset_name, c, trainp,
                                                  gpu_to_use, q, ret_tag, ret_dict, dataset_mode))
                processes.append(process)
                process.start()

            exp_func(func_iter)

    for p in processes:
        p.join()
    with open(ret_filename, 'a') as f:
        for k in ret_dict:
            f.write('\n'.join(ret_dict[k]))
            f.write('\n')
#
# if __name__ == '__main__':
#     n_run = 1 #TODO
#     #N_GPUS = [0, 1, 2, 3, 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
#     N_GPUS = [0, 1, 2, 0,1,2, 3, 4, 3, 4]
#     #N_GPUS = [0]
#     man = Manager()
#     q = man.Queue(len(N_GPUS))
#     for g in N_GPUS:
#         q.put(str(g))
#
#     experiments_list = [
#         #(load_reuters_with_outliers_general, 'reuters', 5),
#         #(load_20news_with_outliers_general, '20news', 20),
#         #(load_caltech101_with_outliers_general, 'caltech101', 11),
#         #(load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
#         (load_cifar10_with_outliers_general, 'cifar10', 10),
#     ]
#
#     p_list = [
#                ('TEST', 0.1, 'SAME', None),
#                 ('TEST', 0.3, 'SAME', None),
#                ('TEST', 0.5, 'SAME', None),
#                 ('TEST', 0.7, 'SAME', None),
#                ('TEST', 0.9, 'SAME', None),
#               ]
#
#     _Noi = 0.
#     _Lam = 1.
#     _Lamb1 = 2.0
#     _Lamb2 = 0.01
#     _Pca_mode = 'B'
#     _Z_size = 10
#     _ShareAB = True
#     _TrainMode = "NAD"
#     _LossMode = None
#     _Lr = 0.0001
#     _Flatten_size = 128
#     ret_filename = './logs/in_coteaching_{}.log'.format(
#         datetime.now().strftime('%Y-%m-%d-%H%M'))
#     # N_GPUS = [0, 1, 2]
#     for i in range(n_run):
#         for data_load_fn, dataset_name, n_classes in experiments_list:
#             for p in p_list:
#                 run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
#                                         [_MPCA_experiment],
#                                         "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}hist".format(
#                                             _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
#                                             _TrainMode, _LossMode, _Flatten_size,
#                                             _Lr))
#
#     _Pca_mode = 'A'
#     for i in range(n_run):
#         for data_load_fn, dataset_name, n_classes in experiments_list:
#             for p in p_list:
#                 run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
#                                         [_MPCA_experiment],
#                                         "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}hist".format(
#                                             _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
#                                             _TrainMode, _LossMode, _Flatten_size,
#                                             _Lr))


if __name__ == '__main__':
    n_run = 1 #TODO
    #N_GPUS = [0, 1, 2, 3, 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    N_GPUS = [4, 5, 6, 7, 4, 5, 6, 7, 3, 3]
    #N_GPUS = [0]
    man = Manager()
    q = man.Queue(len(N_GPUS))
    for g in N_GPUS:
        q.put(str(g))

    experiments_list = [
        #(load_reuters_with_outliers_general, 'reuters', 5),
        #(load_20news_with_outliers_general, '20news', 20),
        #(load_caltech101_with_outliers_general, 'caltech101', 11),
        #(load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
        (load_cifar10_with_outliers_general, 'cifar10', 10),
    ]

    p_list = [
               ('TEST', 0.1, 'SAME', None),
               ('TEST', 0.3, 'SAME', None),
               ('TEST', 0.5, 'SAME', None),
                ('TEST', 0.7, 'SAME', None),
               ('TEST', 0.9, 'SAME', None),
              ]

    ret_filename = 'logs/MPCA_GT_E3_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    _OP_TYPE = 'RA'
    _Score_Mode = 'pl_mean' #'neg_entropy' # 'pl_mean'
    for i in range(n_run):
        for lr in [0.0001]:
            for l1, l2 in [(0.0002, 0.00001)]:
                _Noi = 0.
                _Lamb1, _Lamb2 = l1, l2
                _Pca_mode = 'B'
                _Z_size = 64
                _ShareAB = True
                _Proj_mode = 'batch'
                _Lr = lr
                for data_load_fn, dataset_name, n_classes in experiments_list:
                    for p in p_list:
                        run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                                [_MPCA_GT_experiment],
                                                "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}E3outlier".format(
                                                    _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
                                                    _Lr))

'''
if __name__ == '__main__':
    if args.clear_results:
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
        print("Clear previous results")
    n_run = 5 #TODO
    #N_GPUS = [0, 1, 2, 3, 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    N_GPUS = [ 3, 4, 5, 6, 7,  3, 4, 5, 6, 7]
    #N_GPUS = [0]
    man = Manager()
    q = man.Queue(len(N_GPUS))
    for g in N_GPUS:
        q.put(str(g))

    experiments_list = [
        (load_reuters_with_outliers_general, 'reuters', 5),
        (load_20news_with_outliers_general, '20news', 20),
        # (load_mnist_with_outliers_general, 'mnist', 10),
        # (load_tinyimagenet_with_outliers_general, 'tinyimagenet', 10),
        (load_caltech101_with_outliers_general, 'caltech101', 11),
        # (load_fashion_mnist_with_outliers_rsrae, 'fashion-mnist-rsrae', 10),
        #(load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
         (load_cifar10_with_outliers_general, 'cifar10', 10),
        # (load_cifar100_with_outliers_general, 'cifar100', 20),
        #(load_svhn_with_outliers, 'svhn', 10),
    ]

    # p_list = [0.05, 0.1, 0.15, 0.2, 0.25] TODO
    p_list = [#('SINGLE',  0, 'ALL', None),
               ('TEST', 0.1, 'SAME', None),
              #('PERCENT', 0.1, 'PERCENT', 0.1),
              #('PERCENT', 0.2, 'SAME', None),
                ('TEST', 0.3, 'SAME', None),
               ('TEST', 0.5, 'SAME', None),
                ('TEST', 0.7, 'SAME', None),
               ('TEST', 0.9, 'SAME', None),
              ]

    # _Noi = 0.
    # _Lamb1 = 1.01
    # _Lamb2 = 0.01
    # _Op_Type = "AD"
    #
    # ret_filename = 'logs/benchmarks_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    #
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_rsrae_experiment],
    #                                     "fix1")

    # _Noi = 0.
    # _Lam = 1.
    # _Lamb1 = 2.0
    # _Lamb2 = 0.01
    # _Pca_mode = 'B'
    # _Z_size = 10
    # _ShareAB = True
    # _TrainMode = "NAD"
    # _LossMode = None
    # _Lr = 0.0001
    # _Flatten_size = 128
    # ret_filename = 'logs/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_MPCA_experiment],
    #                                     "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}BackboneV6".format(
    #                                         _Noi,_Lamb1,_Lamb2,_Pca_mode, _Z_size,
    #                                         _TrainMode, _LossMode, _Flatten_size,
    #                                     _Lr))
    # _Pca_mode = 'A'
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_MPCA_experiment],
    #                                     "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}BackboneV6".format(
    #                                         _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
    #                                         _TrainMode, _LossMode, _Flatten_size,
    #                                         _Lr))

    ret_filename = 'logs/MPCA_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    for a, b, c in itertools.product((1.,0.), ((2.1, 0.1),), ('A', 'B')):
        _Noi = a
        _Lamb1, _Lamb2 = b
        _Pca_mode = c
        _Z_size = 10
        _ShareAB = True
        for i in range(n_run):
            for data_load_fn, dataset_name, n_classes in experiments_list:
                for p in p_list:
                    run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                            [_MPCA_experiment],
                                            "noi{}lam({},{})mode{}z{}shareAB{}".format(
                                                _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,_ShareAB))

    ret_filename = 'logs/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    for a, b, c in itertools.product((1., 0.,), (1., 0.), ('exchange', 'neg', 'None')):
        _Noi = a
        _Lam= b
        _Mode = c
        for i in range(n_run):
            for data_load_fn, dataset_name, n_classes in experiments_list:
                for p in p_list:
                    run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                            [_estrsrae_experiment],
                                            "noi{}lam({})mode:{}".format(
                                                _Noi, _Lam, _Mode))

    # ret_filename = 'logs/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for i in range(n_run):
    #     for a, b, c, d, e, f, g in itertools.product((0.,), ((0.0001, 0.0001), ), ('B',),
    #                                        (True,), ("batch",), (256,), (0.002, )):
    #         _Noi = a
    #         _Lamb1,_Lamb2 = b
    #         _Pca_mode = c
    #         _Z_size = f
    #         _ShareAB = d
    #         _Proj_mode = e
    #         _Lr = g
    #         for data_load_fn, dataset_name, n_classes in experiments_list:
    #             for p in p_list:
    #                 run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                         [_MPCA_GT_experiment],
    #                                         "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}".format(
    #                                             _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode, _Lr))


    # for i in range(n_run):
    #     for a, b, c, d, e, f, g in itertools.product((0.,), ((0.0001, 0.0001), (0.00001, 0.00001),), ('B',),
    #                                        (True, False, ), ('batch', "all",), (256,128,), (0.002, 0.003,)):
    #         _Noi = a
    #         _Lamb1,_Lamb2 = b
    #         _Pca_mode = c
    #         _Z_size = f
    #         _ShareAB = d
    #         _Proj_mode = e
    #         _Lr = g
    #         for data_load_fn, dataset_name, n_classes in experiments_list:
    #             for p in p_list:
    #                 run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                         [_MPCA_GT_experiment],
    #                                         "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}".format(
    #                                             _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode, _Lr))

    """
    ret_filename = 'logs/benchmarks_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))

    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [],
                                        "")#"score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))
    """
    """    ret_filename = 'logs/benchmarks_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))

    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_dagmm_experiment],
                                        "")#"score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))
    """
    # ret_filename = 'logs/MPCA_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for a, b, c, d, e, f, g in itertools.product((1., 0.), ((2.0, 0.1), (1., 0.1)), ('A', 'B'),
    #                                  (10, ), ("NAD", ), (0.00005, 0.0001, 0.00025), (128, 64)):
    #     _Noi = a
    #     _Lamb1, _Lamb2 = b
    #     _Pca_mode = c
    #     _Z_size = d
    #     _ShareAB = True
    #     _TrainMode = e
    #     _LossMode = None
    #     _Lr = f
    #     _Flatten_size = g
    #     for i in range(n_run):
    #         for data_load_fn, dataset_name, n_classes in experiments_list:
    #             for p in p_list:
    #                 run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                         [_MPCA_experiment],
    #                                         "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}lr{}fsize{}BackboneV6".format(
    #                                             _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _TrainMode, _LossMode, _Lr, _Flatten_size))
    #

    # baseline for GEOM and E3outlier

    # SCORE_MODE = 'neg_entropy'
    # OP_TYPE = 'RA+IA+PR'
    # ret_filename = 'logs/benchmarks_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    #
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_e3outlier_pytorch_experiment],
    #                                     "score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))
    #
    # SCORE_MODE = 'pl_mean'
    # OP_TYPE = 'RA'
    #
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_e3outlier_pytorch_experiment],
    #                                     "score:{}back:{}transform:{}".format(SCORE_MODE, BACKEND, OP_TYPE))

    # MPCA Linear
    # cal11 fsize 32 lr 0.00005 zsize: 10
    # _Noi = 0.
    # _Lam = 1.
    # _Lamb1 = 2.0
    # _Lamb2 = 0.01
    # _Pca_mode = 'B'
    # _Z_size = 10
    # _ShareAB = True
    # _TrainMode = "NAD"
    # _LossMode = None
    # _Lr = 0.00005
    # _Flatten_size = 32
    # ret_filename = 'logs/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_MPCA_experiment],
    #                                     "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}BackboneV6".format(
    #                                         _Noi,_Lamb1,_Lamb2,_Pca_mode, _Z_size,
    #                                         _TrainMode, _LossMode, _Flatten_size,
    #                                     _Lr))
    # _Pca_mode = 'A'
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_MPCA_experiment],
    #                                     "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}BackboneV6".format(
    #                                         _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
    #                                         _TrainMode, _LossMode, _Flatten_size,
    #                                         _Lr))

    # cal 11 : lr 0.0005  0.0002, 0.00001 lr: 0.0002
    # cifar10: lr 0.00005 0.0002, 0.00001
    # MPCA GT
    # ret_filename = 'logs/MPCA_GT_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for i in range(n_run):
    #     for lr in [0.0002]:
    #         for l1, l2 in [(0.0002, 0.00001)]:
    #             _Noi = 0.
    #             _Lamb1,_Lamb2 = l1, l2
    #             _Pca_mode = 'B'
    #             _Z_size = 64
    #             _ShareAB = True
    #             _Proj_mode = 'batch'
    #             _Lr = lr
    #             for data_load_fn, dataset_name, n_classes in experiments_list:
    #                 for p in p_list:
    #                     run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                             [_MPCA_GT_experiment],
    #                                             "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}".format(
    #                                                 _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode, _Lr))

    # MPCA simple compare with reconstruction
    #ret_filename = 'logs/MPCA_GT_SIM_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # for i in range(n_run):
    #     for lr in [0.0002]:
    #         for l1, l2 in [(0.0002, 0.)]:
    #             _Noi = 0.
    #             _Lamb1, _Lamb2 = l1, l2
    #             _Pca_mode = 'B'
    #             _Z_size = 64
    #             _ShareAB = False
    #             _Proj_mode = 'batch'
    #             _Lr = lr
    #             for data_load_fn, dataset_name, n_classes in experiments_list:
    #                 for p in p_list:
    #                     run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                             [_MPCA_GT_experiment],
    #                                             "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}".format(
    #                                                 _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
    #                                                 _Lr))

    # MPCA_GT_IN_E3outlier
    # ret_filename = 'logs/MPCA_GT_E3_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    # _OP_TYPE = 'RA+IA+PR'
    # _Score_Mode = 'neg_entropy' # 'pl_mean'
    # for i in range(n_run):
    #     for lr in [0.0002]:
    #         for l1, l2 in [(0.000, 0.0000)]:
    #             _Noi = 0.
    #             _Lamb1, _Lamb2 = l1, l2
    #             _Pca_mode = 'B'
    #             _Z_size = 64
    #             _ShareAB = True
    #             _Proj_mode = 'batch'
    #             _Lr = lr
    #             for data_load_fn, dataset_name, n_classes in experiments_list:
    #                 for p in p_list:
    #                     run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                             [_MPCA_GT_experiment],
    #                                             "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}E3outlier".format(
    #                                                 _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
    #                                                 _Lr))




    #直方图实验
    # MPCA Linear
    # cal11 fsize 32 lr 0.00005 zsize: 10
    _Noi = 0.
    _Lam = 1.
    _Lamb1 = 2.0
    _Lamb2 = 0.01
    _Pca_mode = 'B'
    _Z_size = 10
    _ShareAB = True
    _TrainMode = "NAD"
    _LossMode = None
    _Lr = 0.00005
    _Flatten_size = 32
    ret_filename = '/data/linjinghuang/MPCA_results/log_tmp/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    RESULTS_DIR = '/data/linjinghuang/MPCA_results/log_scores'
    # N_GPUS = [0, 1, 2]
    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_MPCA_experiment],
                                        "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}hist".format(
                                            _Noi,_Lamb1,_Lamb2,_Pca_mode, _Z_size,
                                            _TrainMode, _LossMode, _Flatten_size,
                                        _Lr))

    _Pca_mode = 'A'
    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_MPCA_experiment],
                                        "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}BackboneV6".format(
                                            _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
                                            _TrainMode, _LossMode, _Flatten_size,
                                            _Lr))
                
'''

