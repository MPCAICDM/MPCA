import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import math

def get_data(filepath):
    ret_dic = []
    print(filepath)
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                if line.startswith('mean_pos'):
                    # print(line)
                    pp = [float(x.split(':')[1][:10]) for x in line.strip().split(' ')[:4]]
                    #print(pp)
                    #print(line.strip().split(',')[:4])
                    ret_dic.append(pp)
            except:
                print(line)
                ret_dic.append([0.,0., 0.,0.])

    return ret_dic


#_FileNames = ['gt', 'cae', 'mpca', 'mpca_gt']
_FileNames = ['gt', 'mpca_gt', 'mpca_gt','mpca_gt']
_FileNames = ['recons_error_{}_2'.format(f) for f in _FileNames]

def draw_error_curve(datas, idx):
    print(idx)
    factor = 1 if idx == 0 or idx == 3 else 10
    print(datas)
    start_idx = 12
    if idx >= 2:
        datas = datas[::4]
        print(datas)
    if idx == 3 or idx == 0:
        datas = datas[start_idx:]# datas[1:]
        datas[-1] = [(a + b) / 2 for a, b in zip(datas[-3],datas[-2])]
    print(datas)
    for i in range(2):
        #print(datas[0])
        means = [abs(p[i]) for p in datas]
        stds = [p[i+2] for p in datas]

        upper = [m + s for m, s in zip(means, stds)]
        lower = [m - s for m, s in zip(means, stds)]
        labels = ['Inliers', 'Outliers']
        color = ['g', 'r']
        plt.plot(np.arange(start_idx, len(means) * factor + start_idx, factor).astype(dtype=np.str), means, c=color[i],label=labels[i])

        plt.fill_between(np.arange(start_idx, len(means) * factor + start_idx, factor).astype(dtype=np.str),
                         upper,
                         lower,
                         color=color[i],
                         alpha=0.2)
    #plt.legend(prop={'size': 15})
    plt.tick_params(labelsize=13)
    names = ['GEOM', 'AE/CAE', 'AE-LFR', 'GT-LFR']
    plt.xlabel(names[idx], fontsize=20)
    #plt.ylabel('Scores', fontsize=20)
    if idx == 1:
        plt.ylim((0, 100))

    plt.savefig('../figures/{}.pdf'.format(_FileNames[idx]),bbox_inches = 'tight')
    plt.show()

def draw_error_curve_merge(datas1, datas2):
    factor = 1
    datas1 = datas1[1:]
    datas2 = datas2[::4][1:]
    for datas, color in [(datas1, ['lightgreen', 'lightcyan']), (datas2, ['g', 'r'])]:
        for i in range(2):
            #print(datas[0])
            means = [abs(p[i]) for p in datas]
            stds = [p[i+2] for p in datas]

            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            labels = ['Inliers', 'Outliers']
            plt.plot(np.arange(1, len(means) * factor + 1, factor), means, c=color[i],label=labels[i])

            plt.fill_between(np.arange(1, len(means) * factor + 1, factor),
                             upper,
                             lower,
                             color=color[i],
                             alpha=0.2)
    plt.legend(prop={'size': 15})
    plt.tick_params(labelsize=13)
    names = ['GEOM', 'AE/CAE', 'AE-LFR', 'GT-LFR']
    plt.xlabel('Mix', fontsize=20)
    #plt.ylabel('Scores', fontsize=20)
    # if idx == 1:
    #     plt.ylim((0, 100))

    plt.savefig('../figures/{}.pdf'.format('Mix'),bbox_inches = 'tight')
    plt.show()
    print(datas1[-1])
    print(datas2[-1])

#draw_error_curve_merge(get_data('/Users/linjinghuang/data/MPCA/log_score_mean/{}.log'.format(_FileNames[0])),
#                       get_data('/Users/linjinghuang/data/MPCA/log_score_mean/{}.log'.format(_FileNames[1])))

for i in range(2):
    draw_error_curve(get_data('/Users/linjinghuang/data/MPCA/log_score_mean/{}.log'.format(_FileNames[i])),
                 [0, 3][i])