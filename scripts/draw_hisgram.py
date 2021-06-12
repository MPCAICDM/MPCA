import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np



def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'mnist':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'svhn':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'reuters': [str(i) for i in range(5)],
        '20news': [str(i) for i in range(20)],
        'caltech101': [str(i) for i in range(11)],
    }

    return ind_to_name[dataset_name][index]


def draw_hist(scores, labels, tag="none"):
    plt.figure(figsize=(3, 2))
    #plt.title(tag)
    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]
    #plt.hist(scores_pos, bins=64, facecolor='g', edgecolor='g')
    #plt.hist(scores_neg, bins=64, fac)
    sns.distplot(scores_pos, bins=64, kde=False, hist_kws={'color': 'green'}, label='inliers')
    sns.distplot(scores_neg, bins=64, kde=False, hist_kws={'color': 'red'}, label='outliers')
    #plt.xlabel('Reconstruction error')
    #plt.ylabel('Number of data')
    # plt.subplots_adjust(right=0.7)
    #plt.savefig('../figures/%s_%s.png' % (k, score_type))
    # plt.subplots_adjust(right=0.7)
    plt.savefig('/Users/linjinghuang/data/MPCA/imgs/%s.pdf' % tag[:-4],bbox_inches = 'tight')
    #plt.show()

def main_ae():
    items = [
        ('/Users/linjinghuang/data/MPCA/log_hist/caltech101/caltech101_ae_backbone_0.5_2_0.npz',
         'cae_caltech101_2'),
        ('/Users/linjinghuang/data/MPCA/log_hist/fashion-mnist/fashion-mnist_ae_backbone_0.5_sneaker_0.npz',
         'cae_fmnist_sneaker'),
        ('/Users/linjinghuang/data/MPCA/log_hist/fashion-mnist/fashion-mnist_e3outlier_0.5_sneaker_0.npz',
         'gt_fmnist_sneaker'),
        ('/Users/linjinghuang/data/MPCA/log_hist/cifar10/cifar10_e3outlier_0.5_frog_0.npz',
         'gt_cifar10_frog')
    ]
    for item in items:
        scores_info = np.load(item[0])
        draw_hist(-scores_info['scores'], scores_info['labels'], item[1])

def main1():
    dset_name = 'caltech101'
    dir_path = '/Users/linjinghuang/data/MPCA/%s' % dset_name
    datasets_classnum = {
        "reuters": 5,
        "20news": 20,
        "tinyimagenet": 10,
        "caltech101": 11,
        "fashion-mnist": 10,
        "cifar10": 10,
        "mnist": 10
    }
    # for filename in os.listdir(dir_path):
    #     print(filename)
    #     scores_info = np.load(os.path.join(dir_path, filename))
    #     print(scores_info['roc_auc'])
    #     tags = filename[:-4].split('_')
    #     dataset_name, method_mode, c, class_name, run_id = tuple(tags)
    #     draw_hist(scores_info['scores'], scores_info['labels'], filename)
    #     break
    for class_id in range(datasets_classnum[dset_name]):
        for c in [0.1 + 0.2 * i for i in range(5)]:
            f = 'B'
            for t in ['A', 'B']:
                tags = [dset_name, 'mpca-%s-%s' % (f, t),
                        '%.1f' % c, get_class_name_from_index(class_id, dset_name), '0']
                filename = '_'.join(tags) + '.npz'
                scores_info = np.load(os.path.join(dir_path, filename))
                #     print(scores_info['roc_auc'])

                if 'gt' in filename:
                    #draw_hist(1 - np.exp(scores_info['scores']), scores_info['labels'], filename)
                    draw_hist(-scores_info['scores'], scores_info['labels'], filename)
                else:
                    draw_hist(-scores_info['scores'], scores_info['labels'], filename)


main_ae()