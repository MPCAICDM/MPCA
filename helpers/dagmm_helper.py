from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.ae_backbone_keras import Encoder_keras, Decoder_keras, Decoder_Linear_keras, Encoder_Linear_keras
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from loss_functions.drae_loss import DRAELossAutograd
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from models import dagmm
import keras
import numpy as np

class DAGMMHelper(TrainTestHelper):
    def __init__(self, n_channels, h, w, y_test, *args, **kwargs):
        super(DAGMMHelper, self).__init__(*args, **kwargs)
        self.method_tag = "dagmm"

        self.n_channels = n_channels
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist',
                                 'tinyimagenet', 'cifar10']:
            self.encoder = Encoder_keras(input_shape=(h, w, n_channels), representation_dim=5)
            print(self.encoder.summary())
            self.decoder = Decoder_keras(input_shape=(h, w, n_channels), representation_dim=self.encoder.output_shape[-1])
            print(self.decoder.summary())
        else:
            self.encoder = Encoder_Linear_keras(input_shape=(n_channels,), representation_dim=5)
            print(self.encoder.summary())
            self.decoder = Decoder_Linear_keras(input_shape=(5,), representation_dim=n_channels)
            print(self.decoder.summary())

            #self.model = AE_backbone(input_shape=n_channels,hidden_layer_sizes=[32, 64, 128],bn=False).cuda()
        #self.batch_size = 128
        n_components = 3
        self.estimation = Sequential([Dense(64, activation='tanh', input_dim=self.encoder.output_shape[-1] + 2), Dropout(0.5),
                                      Dense(10, activation='tanh'), Dropout(0.5),
                                      Dense(n_components, activation='softmax')]
                                     )
        lambda_diag = 0.005
        lambda_energy = 0.1
        dagmm_mdl = dagmm.create_dagmm_model(self.encoder, self.decoder, self.estimation, lambda_diag)
        optimizer = keras.optimizers.Adam(lr=1e-4)  # default config
        dagmm_mdl.compile(optimizer, ['mse', lambda y_true, y_pred: lambda_energy * y_pred])
        self.dagmm_mdl = dagmm_mdl
        self.y_test = y_test
        print(self.dagmm_mdl.summary())


    def train(self):
        self.dagmm_mdl.fit(x=self.trainset, y=[self.trainset, np.zeros((len(self.trainset), 1))],  # second y is dummy
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(self.trainset, [self.trainset, np.zeros((len(self.trainset), 1))]),
                      # verbose=0
                      )


    def compute_scores(self):
        energy_mdl = Model(self.dagmm_mdl.input, self.dagmm_mdl.output[-1])
        scores = -energy_mdl.predict(self.testset, self.batch_size)
        scores = scores.flatten()
        if not np.all(np.isfinite(scores)):
            min_finite = np.min(scores[np.isfinite(scores)])
            scores[~np.isfinite(scores)] = min_finite - 1
        labels = self.y_test.flatten()
        return scores, labels

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)