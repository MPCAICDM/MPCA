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
from helpers.cae_rsrae_helper import AERSEHelper
from helpers.rdae_helper import RDAEHelper

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
RESULTS_DIR = args.results_dir
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

# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
#
# KTF.set_session(sess)

def _cae_baseline_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
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
    cae_helper = AERSEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=10, is_save_train=True,
                           h=h, w=w, n_channels=n_channels,
                           epochs=1000, hidden_layer_sizes=[32, 64, 128])
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

def _rdae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
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
    cae_helper = RDAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=100, is_save_train=False,
                            epochs=1000,
                            n_channels=n_channels, h=h, w=w, x_train=x_train, y_train=y_train)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _e3outlier_pytorch_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                  gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outlierHelper(trainset=None, testset=None, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=10, n_channels=n_channels,OP_TYPE=OP_TYPE,BACKEND=BACKEND,
                                 SCORE_MODE=SCORE_MODE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
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
        np.random.seed(run_idx)
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



if __name__ == '__main__':
    if args.clear_results:
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
        print("Clear previous results")
    n_run = 5 #TODO
    N_GPUS = [0, 1, 2, 3, 0,1,2,3,0,1,2,3]
    #N_GPUS = [0, 1, 2, 3]
    #N_GPUS = [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7]
    #N_GPUS = [0,1]
    man = Manager()
    q = man.Queue(len(N_GPUS))
    for g in N_GPUS:
        q.put(str(g))

    experiments_list = [
        #(load_reuters_with_outliers_general, 'reuters', 5),
        #(load_20news_with_outliers_general, '20news', 20),
        # (load_mnist_with_outliers_general, 'mnist', 10),
        # (load_tinyimagenet_with_outliers_general, 'tinyimagenet', 10),
        #(load_caltech101_with_outliers_general, 'caltech101', 11),
        #(load_fashion_mnist_with_outliers_rsrae, 'fashion-mnist-rsrae', 10),
        (load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
        #(load_cifar10_with_outliers_general, 'cifar10', 10),
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



    #ret_filename = 'logs/benchmarks_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))

    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_drae_experiment],
    #                                     "")#"score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))

    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_cae_baseline_experiment, _drae_experiment],
    #                                     "")#"score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))


    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_rdae_experiment],
    #                                     "")#"score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))

    # baseline for GEOM and E3outlier

    SCORE_MODE = 'neg_entropy'
    OP_TYPE = 'RA+IA+PR'
    ret_filename = 'logs/E3outlier_GEOM_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))

    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_e3outlier_pytorch_experiment],
                                        "score:{}back:{}transform:{}".format(SCORE_MODE,BACKEND,OP_TYPE))

    SCORE_MODE = 'pl_mean'
    OP_TYPE = 'RA'

    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes in experiments_list:
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_e3outlier_pytorch_experiment],
                                        "score:{}back:{}transform:{}".format(SCORE_MODE, BACKEND, OP_TYPE))