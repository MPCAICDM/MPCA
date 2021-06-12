from glob import glob
import os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import normalize as nmlz
from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from keras.backend import cast_to_floatx
from torchvision.datasets import SVHN
import matplotlib.pyplot as plt
import math
import pickle


def resize_and_crop_image(input_file, output_side_length, greyscale=False):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    height_offset = (new_height - output_side_length) // 2
    width_offset = (new_width - output_side_length) // 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    assert cropped_img.shape[:2] == (output_side_length, output_side_length)
    return cropped_img


def normalize_minus1_1(data):
    return 2*(data/255.) - 1


def get_channels_axis():
    import keras
    idf = keras.backend.image_data_format()
    if idf == 'channels_first':
        return 1
    assert idf == 'channels_last'
    return 3

def load_fashion_mnist_nopad():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    siz = 28
    X_train = np.reshape(X_train, (-1, siz*siz))
    X_train = nmlz(X_train)
    X_train = np.reshape(X_train, (-1, siz, siz, 1))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    X_test = np.reshape(X_test, (-1, siz * siz))
    X_test = nmlz(X_test)
    X_test = np.reshape(X_test, (-1, siz, siz, 1))
    return (X_train, y_train), (X_test, y_test)

def load_fashion_mnist(padding=True):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    siz = 28
    if padding:
        X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')
        X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')
        siz = 32
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))

    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_mnist(padding=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    siz = 28
    if padding:
        X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')
        X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')
        siz = 32
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)

def load_caltech101(padding=False, data_path='data/caltech101.data'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X_test = data["X"]
    y_test = data["y"]
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (None, None), (X_test, y_test)

def load_tinyimagenet(padding=False, data_path='data/tinyimagenet.data'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X_test = data["X"]
    y_test = data["y"]
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (None, None), (X_test, y_test)

def load_20news(padding=False, data_path='data/20news.data'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X_test = np.array(data["X"], dtype=np.float32)
    y_test = np.array(data["y"])
    return (None, None), (X_test, y_test)

def load_reuters(padding=False, data_path='data/reuters.data'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X_test = np.array(data["X"], dtype=np.float32)
    y_test = np.array(data["y"])
    return (None, None), (X_test, y_test)

def load_cifar10(padding=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    siz = 32
    return (X_train, y_train), (X_test, y_test)


def load_cifar100(padding=True, label_mode='coarse'):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    siz = 32
    return (X_train, y_train), (X_test, y_test)

def load_svhn(padding=True, is_nmlz=False, data_dir='.SVHN_data/'):
    img_train_data = SVHN(root=data_dir, split='train', download=True)
    img_test_data = SVHN(root=data_dir, split='test', download=True)
    X_train = img_train_data.data.transpose((0, 2, 3, 1))
    y_train = img_train_data.labels
    X_test = img_test_data.data.transpose((0, 2, 3, 1))
    y_test = img_test_data.labels
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    siz = 32
    if is_nmlz:
        X_train = np.reshape(X_train, (-1, siz*siz*3))
        X_train = nmlz(X_train)
        X_train = np.reshape(X_train, (-1, siz, siz, 3))
        X_test = np.reshape(X_test, (-1, siz * siz*3))
        X_test = nmlz(X_test)
        X_test = np.reshape(X_test, (-1, siz, siz, 3))
    return (X_train, y_train), (X_test, y_test)

def save_false_sample(scores, test_data, labels, file_path="/data/ljh/anodect/results/test.pmg",
                      topk=10):
    scores = scores.flatten()
    labels = labels.flatten()
    assert len(scores) == len(labels) and len(labels) == len(test_data)
    ids = list(range(len(labels)))
    ids = sorted(ids, key=lambda x: -scores[x])
    colsize = 4
    low_neg = []
    low_pos = []
    for i in ids:
        #print(scores[i])
        if labels[i] != 1:
            low_neg.append(i)
    for i in reversed(ids):
        if labels[i] == 1:
            low_pos.append(i)
    low_neg = low_neg[:topk]
    low_pos = low_pos[:topk]
    rowsize = math.ceil(topk/colsize) * 2
    plt.figure(figsize=(colsize * 2, rowsize * 2.2))
    for idx, sid in enumerate(low_neg):
        plt.subplot(rowsize, colsize, idx + 1)
        plt.imshow(test_data[sid], cmap='gray')
        #print(test_data[sid].size)
        #print(test_data[sid].getpixel((10,12)))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("s=%0.2f" % scores[sid])
    for idx, sid in enumerate(low_pos):
        plt.subplot(rowsize, colsize, math.ceil(topk/colsize) * colsize + idx + 1)
        plt.imshow(test_data[sid], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("s=%0.2f" % scores[sid])
    plt.suptitle(file_path)
    #plt.legend()
    plt.savefig(file_path)
    plt.show()


def show_roc_pr_curve_data(scores, labels):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom))
    return roc_auc, pr_auc_norm, pr_auc_anom


def show_avg_scores(scores, labels):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    mean_pos = np.mean(scores_pos)
    mean_neg = np.mean(scores_neg)

    std_pos = np.std(scores_pos)
    std_neg = np.std(scores_neg)

    #print("mean_pos:{}, mean_neg:{} std_pos:{} std_neg:{}".format(mean_pos, mean_neg, std_pos, std_neg))
    return mean_pos, mean_neg, std_pos, std_neg


def save_roc_pr_curve_data(scores, labels, file_path, mode="SAVE"):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    if mode != "SAVE":
        print(roc_auc)
        return

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        scores=scores, labels=labels,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def create_cats_vs_dogs_npz(cats_vs_dogs_path='./'):
    labels = ['cat', 'dog']
    label_to_y_dict = {l: i for i, l in enumerate(labels)}

    def _load_from_dir(dir_name):
        glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
        imgs_paths = glob(glob_path)
        images = [resize_and_crop_image(p, 64) for p in imgs_paths]
        x = np.stack(images)
        y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
        y = np.array(y)
        return x, y

    x_train, y_train = _load_from_dir('train')
    x_test, y_test = _load_from_dir('test')

    np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)


def load_cats_vs_dogs(cats_vs_dogs_path='./'):
    npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
    x_train = normalize_minus1_1(cast_to_floatx(npz_file['x_train']))
    y_train = npz_file['y_train']
    x_test = normalize_minus1_1(cast_to_floatx(npz_file['x_test']))
    y_test = npz_file['y_test']

    return (x_train, y_train), (x_test, y_test)


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


def modify_inf(result):
    # change nan to zero
    result[result != result] = 0

    # change inf to 10**20
    rsize = len(result)

    for i in range(0, rsize):

        if np.isinf(result[i]):
            if result[i] > 0:
                result[i] = +10 ** 20
            else:
                result[i] = -10 ** 20
    return result


def normalize(samples, min, max):
    # type: (np.ndarray, float, float) -> np.ndarray
    """
    Normalize scores as in Eq. 10

    :param samples: the scores to be normalized.
    :param min: the minimum of the desired scores.
    :param max: the maximum of the desired scores.
    :return: the normalized scores
    """

    if (max - min) == 0:
        result = samples
    else:
        result = (samples - min) / (max - min)

    # result =  (samples - min) / (max - min)

    return result


def novelty_score(sample_llk_norm, sample_rec_norm):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Computes the normalized novelty score given likelihood scores, reconstruction scores
    and normalization coefficients (Eq. 9-10).
    :param sample_llk_norm: array of (normalized) log-likelihood scores.
    :param sample_rec_norm: array of (normalized) reconstruction scores.
    :return: array of novelty scores.
    """

    # Sum
    ns = sample_llk_norm + sample_rec_norm

    return ns
