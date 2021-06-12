import argparse
import os
import re
import numpy as np
from utils import get_class_name_from_index
import matplotlib.pyplot as plt


def draw_roc_curve(fpr, tpr, roc_thresholds, roc_auc, title="None"):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area=%0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend("lower right")
    plt.show()

def get_filenames(algo_name, results_dir, dataset_name, class_name):
    """Returns all files satisfying the patterns."""
    pattern = re.compile(r'{}_{}_{}_[0-9\-]+\.npz'.format(
        dataset_name, algo_name, class_name
    ))
    all_files = os.listdir(os.path.join(results_dir, dataset_name))
    selected = [f for f in all_files if pattern.match(f) is not None]
    return sorted([os.path.join(results_dir, dataset_name, f) for f in selected])


def compute_average_roc_auc(algo_name, results_dir, dataset_name, n_classes):
    results = {}
    avg_results = []
    std_results = []
    for c in range(n_classes):
        class_name = get_class_name_from_index(c, dataset_name)
        filenames = get_filenames(algo_name, results_dir, dataset_name, class_name)
        print(filenames)
        results[class_name] = [np.load(f)['roc_auc'] for f in filenames]
        for f in filenames:
            ret = np.load(f)
            fpr = ret['fpr']
            tpr = ret['tpr']
            threshold = ret['roc_thresholds']
            roc_auc = ret['roc_auc']
            draw_roc_curve(fpr, tpr, threshold, roc_auc, f)

    for k, v in results.items():
        print('{}: {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v)))
        avg_results.append(np.mean(v))
        std_results.append(np.std(v))
    # compute the std of average results over multiple runs
    min_runs = min([len(v) for v in results.values()])
    std_rec = []
    for i in range(min_runs):
        ith_run = [results[get_class_name_from_index(c, dataset_name)][i] for c in range(n_classes)]
        std_rec.append(np.mean(ith_run))
    print('-------------------------------------------')
    print('Average: {:.4f} +- {:.4f}'.format(np.mean(avg_results), np.std(std_rec)))
    print('Formated:')
    for r, s in zip(avg_results, std_results):
        print('{:.4f}~{:.4f}'.format(r, s))
    print('{:.4f}~{:.4f}'.format(np.mean(avg_results), np.std(std_rec)))


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Argument parser for AUROC evaluations.')
    parser.add_argument('--algo_name', type=str, default='e3outlier-0.1')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset', type=str, default='cifar10')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_classes = {
        'cifar10': 10, 'mnist': 10, 'cifar100': 20, 'fashion-mnist': 10, 'svhn': 10,
    }[args.dataset]
    compute_average_roc_auc(args.algo_name, args.results_dir, args.dataset,
                            n_classes)
