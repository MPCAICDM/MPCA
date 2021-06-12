import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import math

def get_data(filepath):
    ret_dic = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                if line.startswith('mean_pos'):
                    # print(line)
                    pp = [float(x.split(':')[1][:10]) for x in line.strip().split(',')[:4]]
                    ret_dic.append(pp)
            except:
                print(line)

    return ret_dic


def draw_error_curve(datas):
    for i in range(2):
        rec_data = [-p[i] for p in datas]
        labels = ['Inliers', 'Outliers']
        color = ['g', 'r']
        plt.plot(np.arange(0, len(rec_data) * 10, 10), rec_data, c=color[i],label=labels[i])
    plt.legend(prop={'size': 15})
    plt.tick_params(labelsize=13)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Reconstruction error', fontsize=20)
    plt.ylim((0, 100))

    plt.savefig('../figures/recons_error.pdf',bbox_inches = 'tight')
    plt.show()

draw_error_curve(get_data('./reconstruction.log'))