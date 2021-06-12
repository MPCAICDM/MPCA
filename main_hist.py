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
from helpers.cae_rsrae_helper import AERSEHelper

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
RESULTS_DIR = '/data/linjinghuang/MPCA_results/log_hist'
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
                           batch_size=128, test_per_epoch=100, is_save_train=True,
                           h=h, w=w, n_channels=n_channels,
                           epochs=1000, hidden_layer_sizes=[32, 64, 128])
    cae_helper.print(ret_tag)
    cae_helper.train()
    cae_helper.test(True)
    cae_helper.save()
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
    e3_helper.save()
    ret_logger[ret_tag] = e3_helper.ret_logger
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
                           batch_size=128, test_per_epoch=10, is_save_train=True,h=h,w=w,
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

def _MPCA_GT_2stage_experiment(x_train, y_train,x_test, y_test, dataset_name, c, abnormal_fraction,
                           gpu_to_use, gpu_q, ret_tag, ret_logger, dataset_mode):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
    n_channels = x_train.shape[get_channels_axis()]
    if _First_stage:
        e3_helper = MPCAGTHelper(trainset=None, testset=None, dataset_name=dataset_name,
                                 single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                                 batch_size=128, test_per_epoch=1, is_save_train=True,
                                 epochs=10, n_channels=n_channels, z_channels=_Z_size, num_works=8,
                                 hidden_layer_sizes=[32, 64, 128], lamb1=0., lamb2=0.,
                                 noise_rate=_Noi * (abnormal_fraction / (1 + abnormal_fraction)),
                                 pca_mode=_Pca_mode, shareAB=_ShareAB, proj_mode=_Proj_mode, lr=_Lr,
                                 SCORE_MODE=_Score_Mode, OP_TYPE=_OP_TYPE)
        e3_helper.print(ret_tag)
        e3_helper.transform_traindata(x_train)
        e3_helper.transform_testdata(x_test, y_test)
        e3_helper.train()
        e3_helper.test(True)
        e3_helper.save(ori_tag=_Pca_mode)
        e3_helper.save_weight()
        ret_logger[ret_tag] = e3_helper.ret_logger
    if _Second_stage:
        # print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
        e3_helper = MPCAGTHelper(trainset=None, testset=None, dataset_name=dataset_name,
                               single_class_ind=c, p=abnormal_fraction, RESULTS_DIR=RESULTS_DIR,
                               batch_size=128, test_per_epoch=1, is_save_train=True,
                               epochs=10,n_channels=n_channels, z_channels=_Z_size,num_works=8,
                                 hidden_layer_sizes=[32, 64, 128], lamb1=_Lamb1, lamb2=_Lamb2,
                                 noise_rate=_Noi*(abnormal_fraction/(1+abnormal_fraction)),
                                pca_mode=_Pca_mode, shareAB=_ShareAB, proj_mode=_Proj_mode, lr=_Lr,
                                 SCORE_MODE=_Score_Mode, OP_TYPE=_OP_TYPE, update_mode='A')
        e3_helper.print(ret_tag)
        e3_helper.load_weight()
        print('load weight')
        e3_helper.transform_traindata(x_train)
        e3_helper.transform_testdata(x_test, y_test)
        e3_helper.train()
        e3_helper.test(True)
        e3_helper.save(ori_tag=_Pca_mode)
        e3_helper.save_weight('second')
        ret_logger[ret_tag] = e3_helper.ret_logger
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

def run_general_experiments(load_dataset_fn, dataset_name, q, n_classes, data_mode, run_idx, experiments_funcs, param_tag):
    ret_dict = man.dict()
    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    train_mode, trainp, test_mode, testp = data_mode
    dataset_mode = "".join([str(i) for i in data_mode])
    print(dataset_mode)
    processes = []

    #experiments_funcs =
    for c in _Classes:
        np.random.seed(run_idx) #
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
    n_run = 1 #TODO
    #N_GPUS = [0, 1, 2, 3,4, 0,1,2,3, 4, 5]
    #N_GPUS = [5,6,7,5,6,7]
    N_GPUS = [1]
    man = Manager()
    q = man.Queue(len(N_GPUS))
    for g in N_GPUS:
        q.put(str(g))

    experiments_list = [
        #(load_reuters_with_outliers_general, 'reuters', 5),
        #(load_20news_with_outliers_general, '20news', 20),
        #(load_caltech101_with_outliers_general, 'caltech101', 11, [2]),
        (load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10, [7]),
        (load_cifar10_with_outliers_general, 'cifar10', 10, [6]),
    ]

    p_list = [
              # ('TEST', 0.1, 'SAME', None),
              #  ('TEST', 0.3, 'SAME', None),
               ('TEST', 0.5, 'SAME', None),
              # ('TEST', 0.7, 'SAME', None),
              # ('TEST', 0.9, 'SAME', None),
              ]

    ret_filename = './logs/hist_{}.log'.format(
        datetime.now().strftime('%Y-%m-%d-%H%M'))
    _Classes = [7]

    SCORE_MODE = 'pl_mean'
    OP_TYPE = 'RA'
    #N_GPUS = [0]
    for i in range(n_run):
        for data_load_fn, dataset_name, n_classes, cc in experiments_list:
            _Classes = cc
            for p in p_list:
                run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
                                        [_e3outlier_pytorch_experiment],
                                        "")


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
    #
    # # N_GPUS = [0, 1, 2]
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_MPCA_experiment],
    #                                     "noi{}lam({},{})mode{}z{}TrainMode{}LossMode{}fsize{}lr{}hist".format(
    #                                         _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size,
    #                                         _TrainMode, _LossMode, _Flatten_size,
    #                                         _Lr))

    #init
    # _OP_TYPE = 'RA'
    # _Score_Mode = 'pl_mean'  # 'neg_entropy' # 'pl_mean'
    # _Noi = 0.
    # _Lamb1, _Lamb2 = 0.0002, 0.00001
    # _Pca_mode = 'B'
    # _Z_size = 20
    # _ShareAB = True
    # _Proj_mode = 'batch'
    # _Lr = 0.0002
    # _First_stage = True
    # _Second_stage = False
    # for data_load_fn, dataset_name, n_classes in experiments_list:
    #     for p in p_list:
    #         run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, 0,
    #                                 [_MPCA_GT_2stage_experiment],
    #                                 "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}init".format(
    #                                     _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
    #                                     _Lr))

    # _First_stage = False
    # _Second_stage = True
    # for data_load_fn, dataset_name, n_classes in experiments_list:
    #     for p in p_list:
    #         run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, 0,
    #                                 [_MPCA_GT_2stage_experiment],
    #                                 "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}second".format(
    #                                     _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
    #                                     _Lr))


    # _OP_TYPE = 'RA'
    # _Score_Mode = 'pl_mean' #'neg_entropy' # 'pl_mean'
    # for i in range(n_run):
    #     for lr in [0.0002]:
    #         for l1, l2 in [(0.0002, 0.00001)]:
    #             _Noi = 0.
    #             _Lamb1, _Lamb2 = l1, l2
    #             _Pca_mode = 'B'
    #             _Z_size = 20
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


    # SCORE_MODE = 'pl_mean'
    # OP_TYPE = 'RA'
    #
    # for i in range(n_run):
    #     for data_load_fn, dataset_name, n_classes in experiments_list:
    #         for p in p_list:
    #             run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
    #                                     [_e3outlier_pytorch_experiment],
    #                                     "score:{}back:{}transform:{}".format(SCORE_MODE, BACKEND, OP_TYPE))
    #







#
# if __name__ == '__main__':
#     n_run = 5 #TODO
#     #N_GPUS = [0, 1, 2, 3, 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
#     N_GPUS = [5, 4, 4, 5,  0, 1, 2, 3, 0, 1, 2, 3]
#     #N_GPUS = [0]
#     man = Manager()
#     q = man.Queue(len(N_GPUS))
#     for g in N_GPUS:
#         q.put(str(g))
#
#     experiments_list = [
#         #(load_reuters_with_outliers_general, 'reuters', 5),
#         #(load_20news_with_outliers_general, '20news', 20),
#         (load_caltech101_with_outliers_general, 'caltech101', 11),
#         #(load_fashion_mnist_with_outliers_general, 'fashion-mnist', 10),
#         #(load_cifar10_with_outliers_general, 'cifar10', 10),
#     ]
#
#     p_list = [
#                #('TEST', 0.1, 'SAME', None),
#                # ('TEST', 0.3, 'SAME', None),
#                #('TEST', 0.5, 'SAME', None),
#                 ('TEST', 0.7, 'SAME', None),
#                #('TEST', 0.9, 'SAME', None),
#               ]
#
#     ret_filename = 'logs/MPCA_GT_E3_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
#     _OP_TYPE = 'RA'
#     _Score_Mode = 'pl_mean' #'neg_entropy' # 'pl_mean'
#     for i in range(n_run):
#         for lr in [0.00015]:
#             for l1, l2 in [(0.0002, 0.00001)]:
#                 _Noi = 0.
#                 _Lamb1, _Lamb2 = l1, l2
#                 _Pca_mode = 'B'
#                 _Z_size = 64
#                 _ShareAB = True
#                 _Proj_mode = 'batch'
#                 _Lr = lr
#                 for data_load_fn, dataset_name, n_classes in experiments_list:
#                     for p in p_list:
#                         run_general_experiments(data_load_fn, dataset_name, q, n_classes, p, i,
#                                                 [_MPCA_GT_experiment],
#                                                 "noi{}lam({},{})mode{}z{}shareABReal{}Projmode{}lr{}E3outlier".format(
#                                                     _Noi, _Lamb1, _Lamb2, _Pca_mode, _Z_size, _ShareAB, _Proj_mode,
#                                                     _Lr))
