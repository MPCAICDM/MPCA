from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data, modify_inf, normalize, novelty_score
from models.MTQ_mnist import MTQ_MNIST
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
import numpy as np
from loss_functions.flow_loss import MTQSOSLoss
from scipy.stats import norm



class MTQHelper(TrainTestHelper):
    def __init__(self, n_channels,h,w,score_norm=True, *args, **kwargs):
        super(MTQHelper, self).__init__(*args, **kwargs)
        self.method_tag = "lsa"

        self.n_channels = n_channels
        #cpd_channels = 100
        #code_length = 64
        lam = 1.
        lr = 0.00001
        self.model = MTQ_MNIST(input_shape=(n_channels, h, w),
                               es_channels=2048,
                               num_blocks=1,
                             hidden_layer_sizes=[32, 64, 128],
                               backbone='LSA',
                               code_length=64).cuda()
        self.code_length = self.model.h_channels

        cudnn.benchmark = True
        self.criterion = MTQSOSLoss(lam=lam).cuda()
        # use adam always
        #self.ae_optimizer = optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder.parameters()), eps=1e-7, weight_decay=0.0005, lr=lr)
        #self.est_optimizer = optim.Adam(self.model.estimator.parameters(), lr=lr, weight_decay=1e-6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        #self.epochs = 250
        self.score_norm = score_norm
        self.print("score_norm:{} lam:{} lr:{}".format(score_norm, lam, lr))


    def train_step(self, x, y=None):
        x = torch.autograd.Variable(x.cuda())
        x_r, z, s, log_jacob_T_inverse = self.model(x)
        loss = self.criterion(x, x_r, s, log_jacob_T_inverse, True)
        #print(loss)
        self.losses.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _eval_quantile(self, s):

    #  from s~N(R^d) to u~(0,1)^d
    #  u_i = phi(s_i), where phi is the cdf of N(0,1)
    # Compute S
    # only for SOS density estimator

        bs = s.shape[0]
        s_dim = s.shape[1]
        s_numpy = s.cpu().numpy()
        q1 = []
        q2 = []
        qinf = []
        u_s = np.zeros((bs , s_dim))
        for i in range(bs):
            # for every sample
            # cdf
            u_si = norm.cdf(s_numpy[i, :])
            u_s[i,:] = u_si
            # Source point in the source uniform distribution
            # u = abs(np.ones((1,s_dim))*0.5-u_s)

            u = abs(0.5 - u_si)

            uq_1 = np.linalg.norm(u, 1)
            uq_2 = np.linalg.norm(u)
            uq_inf = np.linalg.norm(u, np.inf)

            q1.append(-uq_1)
            q2.append(-uq_2)
            qinf.append(-uq_inf)

        return q1, q2, qinf, u_s

    def compute_scores(self):
        self.model.eval()
        sample_q1 = np.zeros(shape=(len(self.testloader.dataset),))
        sample_q2 = np.zeros(shape=(len(self.testloader.dataset),))
        sample_qinf = np.zeros(shape=(len(self.testloader.dataset),))
        sample_u = np.zeros(shape=(len(self.testloader.dataset), self.code_length))

        sample_llk = np.zeros(shape=(len(self.testloader.dataset),))
        sample_nrec = np.zeros(shape=(len(self.testloader.dataset),))
        # true label
        sample_y = np.zeros(shape=(len(self.testloader.dataset),))
        y_test = []
        cc = 0
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r, z, s, log_jacob_T_inverse = self.model(x)
                q1, q2, qinf, u_s = self._eval_quantile(s)
                bs = len(inputs)
                # quantile
                sample_q1[cc:cc + bs] = q1
                sample_q2[cc:cc + bs] = q2
                sample_qinf[cc:cc + bs] = qinf
                # source point
                sample_u[cc:cc + bs] = u_s
                #print(cc, cc+bs)
                _ = self.criterion(x, x_r, s, log_jacob_T_inverse, False)
                sample_nrec[cc:cc + bs] \
                    = - self.criterion.reconstruction_loss.cpu().numpy()
                sample_llk[cc:cc + bs] \
                    = - self.criterion.autoregression_loss.cpu().numpy()

                cc += bs
                y_test.append(labels.data.cpu())
        y_test = torch.cat(y_test, dim=0)
        sample_llk = modify_inf(sample_llk)

        if self.score_norm:
            remax, remin = sample_llk.max(), sample_llk.min()
            aurmax, aurmin = sample_nrec.max(), sample_nrec.min()

            sample_llk = (sample_llk - remin) / (remax - remin + 1e-12)
            sample_nrec = (sample_nrec - aurmin) / (aurmax - aurmin + 1e-12)
        sample_ns = novelty_score(sample_llk, sample_nrec)


        return (sample_ns, sample_llk, sample_nrec, sample_q1, sample_q2, sample_qinf), y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            titles = "sample_ns, sample_llk, sample_nrec, sample_q1, sample_q2, sample_qinf".split(",")
            for i in range(len(titles)):
                self.print(titles[i])
                roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores[i], y_test)
                self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)

        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)