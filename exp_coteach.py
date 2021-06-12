from helpers.coteaching_helper import CoTeachingHelper, CoTeachingResnetHelper
from helpers.e3outlier_pencil_helper import E3outlierPENCILHelper
from helpers.in_coteaching_helper import InCoTeachingHelper, InCoteachingResnetHelper
from helpers.cae_lsa_helper import CAELSAHelper
from helpers.cae_helper import CAEHelper
from helpers.e3outlier_helper import E3outlierHelper, E3outlierV2Helper
from helpers.rsrae_helper import RSRAEHelper

from datetime import datetime
import argparse
import os
from multiprocessing import Manager, freeze_support, Process
import numpy as np
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis, save_false_sample
from outlier_datasets_separate import load_mnist_with_outliers_general, load_cifar10_with_outliers_general, \
    load_cifar100_with_outliers_general, load_fashion_mnist_with_outliers_general
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

def _cae_rsrae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = RSRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=200,input_shape=(h, w, n_channels), z_channels=10,
                             hidden_layer_sizes=[32, 64, 128], lamb1=0., lamb2=0.)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _rsrae_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = RSRAEHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                           single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                           batch_size=128, test_per_epoch=1, is_save_train=True,
                           epochs=200,input_shape=(h, w, n_channels), z_channels=10,
                             hidden_layer_sizes=[32, 64, 128], lamb1=0.1, lamb2=0.1)
    cae_helper.print(ret_tag)
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
                           epochs=70, n_channels=n_channels)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)

def _cae_lsa_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = CAELSAHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=1, is_save_train=True,
                              epochs=70, n_channels=n_channels, h=h, w=w)
    cae_helper.print(ret_tag)
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
                                  gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    e3_helper = E3outlierHelper(trainset=None, testset=None, dataset_name=dataset_name,
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
                              gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    trainset = trainset_pytorch(train_data=x_train, train_labels=y_train, transform=transform_train)
    testset = trainset_pytorch(train_data=x_test, train_labels=y_test, transform=transform_test)
    if get_channels_axis() == 1:
        n_channels, h, w = x_train.shape[1:]
    else:
        h, w, n_channels = x_train.shape[1:]
    cae_helper = InCoTeachingHelper(trainset=trainset, testset=testset, dataset_name=dataset_name,
                              single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                              batch_size=128, test_per_epoch=1, is_save_train=True,
                              epochs=70, n_channels=n_channels, h=h, w=w, group=4)
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    ret_logger[ret_tag] = cae_helper.ret_logger
    gpu_q.put(gpu_to_use)


def _in_coteaching_resnet_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                                    gpu_to_use, gpu_q, ret_tag, ret_logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    print(x_train.shape)
    print(x_test.shape)
    e3_helper = InCoteachingResnetHelper(trainset=None, testset=None, dataset_name=dataset_name,
                                single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                                batch_size=128, test_per_epoch=1, is_save_train=True,
                                epochs=150, n_channels=n_channels, OP_TYPE=OP_TYPE, BACKEND=BACKEND,
                                SCORE_MODE=SCORE_MODE)
    e3_helper.print(ret_tag)
    e3_helper.transform_traindata(x_train)
    e3_helper.transform_testdata(x_test, y_test)
    e3_helper.train()
    e3_helper.test(True)
    ret_logger[ret_tag] = e3_helper.ret_logger
    gpu_q.put(gpu_to_use)


def run_general_experiments(load_dataset_fn, dataset_name, q, n_classes, data_mode, run_idx):
    ret_dict = man.dict()
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    train_mode, trainp, test_mode, testp = data_mode
    dataset_mode = "".join([str(i) for i in data_mode])
    processes = []

    experiments_funcs = [_coteaching_resnet_pytorch_experiment, _e3outlier_pytorch_experiment, _e3outlier_v2_pytorch_experiment]
    for c in list(range(n_classes)):
        np.random.seed(run_idx)
        x_train, y_train, x_test, y_test = load_dataset_fn(c, train_mode, trainp, test_mode, testp)
        for func_iter in experiments_funcs:
            gpu_to_use = q.get()
            # random sampling if the number of data is too large
            if x_train.shape[0] > max_sample_num:
                selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
                x_train = x_train[selected, :]
                y_train = y_train[selected]

            #_in_coteaching_resnet_experiment(x_train, y_train, x_test, y_test, dataset_name, c, trainp,
            #                                 run_idx,dataset_mode, gpu_to_use, q)
            def exp_func(efunc):
                ret_tag = "data_mode:{} dataset:{} tag:{} c:{}".format(data_mode, dataset_name, efunc.__name__, c)
                process = Process(target=efunc, args=(x_train, y_train, x_test, y_test, dataset_name, c, trainp,
                                                  gpu_to_use, q, ret_tag, ret_dict))
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
    n_run = 1 #5 TODO
    #N_GPUS = [0, 1, 2]
    N_GPUS = [0, 1, 2]
    man = Manager()
    q = man.Queue(len(N_GPUS))
    ret_filename = 'logs/in_coteaching_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    for g in N_GPUS:
        q.put(str(g))

    experiments_list = [
        (load_mnist_with_outliers_general, 'mnist', 10),
        (load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
        (load_cifar10_with_outliers_general, 'cifar10', 10),
        #(load_cifar100_with_outliers_general, 'cifar100', 20),
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
    #print(ret_dict)
