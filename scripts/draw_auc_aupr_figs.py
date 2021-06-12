import matplotlib.pyplot as plt
from PIL import Image
import os
#from string import maketrans

abtranstable = str.maketrans('AB','BA')

datasets_classnum = {
    "reuters": 5,
    "20news": 20,
    # "tinyimagenet": 10,
    "caltech101": 11,
    "fashion-mnist": 10,
    "cifar10": 10
}

ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

def draw_all_curve_image(file_path, score_type='AUC'):
    save_data = open(file_path, 'r').readlines()
    save_data = save_data[1:]
    transformed = [x.split(',') for x in save_data]
    # dataset_names = [x for x in datasets_classnum]
    rename_dict = {
        'neg_entropy': 'E3Outlier',
        'pl_mean': 'GEOM',
        '_cae_': 'CAE',
        '_drae_': 'DRAE',
        'rsrae': 'RSRAE'
    }
    line_type_dict = {
        'E3Outlier': ('g', '.'),
        'GEOM': ('gray', 'x'),
        'CAE': ('y', 'P'),
        'DRAE': ('c', 's'),
        'RSRAE': ('m', '*'),
        'BA_AE': ('r', 'D'),
        'BA_GEOM': ('b', '^'),
    }

    show_name_dict = {
        'E3Outlier': '$E^3$Outlier',
        'GEOM': 'GEOM',
        'CAE': 'AE/CAE',
        'DRAE': 'DRAE',
        'RSRAE': 'RSRAE',
        'BA_AE': 'AE-LFR',
        'BA_GEOM': 'GT-LFR',
    }

    method_names = [x[0] for x in transformed]
    transformed = [x[1:] for x in transformed]

    for k in datasets_classnum:
        # plt.title('{}'.format(k))
        res = []

        scores = [x[:5] for x in transformed]
        transformed = [x[5:] for x in transformed]

        for name, scs in zip(method_names, scores):
            if '-' not in scs:
                res.append((name, [float(x.strip()) for x in scs]))

        if score_type == 'AUPR':
            plt.figure(figsize=(5, 6), dpi=200)
        else:
            plt.figure(figsize=(5, 7), dpi=200)
        for i in range(len(res)):
            # print(res[i][0])
            label_name = str(res[i][0])
            #print(label_name, res[i][1])
            print(score_type, k, label_name.translate(abtranstable), ' '.join(['& %s' % x for x in res[i][1]]))
            if label_name in line_type_dict:
                plt.plot(ratios, res[i][1],
                         label=show_name_dict[label_name],
                         marker=line_type_dict[label_name][1],
                         color=line_type_dict[label_name][0])
        #plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        if k == 'cifar10' or True:
            if score_type == 'AUROC':
                #pass
                plt.legend(loc=1,ncol=2)
            else:
                #pass
                plt.legend(loc=4,ncol=2)
        plt.xlabel('Ratio of outliers/inliers c', fontsize=18)
        plt.ylabel(score_type, fontsize=19)
        plt.xlim((0, 1))
        plt.xticks(ratios)
        if score_type == 'AUROC':
            if 'caltech101' == k:
                plt.ylim((0.4, 1))
                plt.yticks([x * 0.1 for x in range(4, 11)])
            else:
                plt.ylim((0.4, 1))
                plt.yticks([x * 0.1 for x in range(4, 11)])
        else:
            plt.ylim((0, 0.9))
            plt.yticks([x * 0.1 for x in range(10)])
        plt.grid(linestyle='--')
        # plt.subplots_adjust(left=0.01)
        plt.savefig('../figures/%s_%s.pdf' % (k, score_type),bbox_inches = 'tight')
        # plt.subplots_adjust(right=0.7)
        plt.show()


draw_all_curve_image('./AUROC.csv', 'AUROC')
draw_all_curve_image('./AUPR.csv', 'AUPR')