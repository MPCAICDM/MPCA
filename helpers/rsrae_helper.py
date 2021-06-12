from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.rsrae import RSRAE, RSRAE_Linear
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from loss_functions.rsr_loss import RSRLoss
import torch.nn.functional as F

class RSRAEHelper(TrainTestHelper):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels,noise_rate,op_type,
                 lamb1, lamb2, *args, **kwargs):
        super(RSRAEHelper, self).__init__(*args, **kwargs)
        self.method_tag = "rsrae"

        self.input_shape = input_shape
        self.print(input_shape)
        #self.model = RSRAE(input_shape, hidden_layer_sizes, z_channels).cuda()
        lr = 0.00025
        if self.dataset_name in ['caltech101', 'fashion-mnist-rsrae', 'fashion-mnist']:
            self.model = RSRAE(input_shape=input_shape, z_channels=10,
                                   hidden_layer_sizes=[32, 64, 128]).cuda()
        else:
            self.model = RSRAE_Linear(input_shape=input_shape[2], z_channels=32,
                                         hidden_layer_sizes=[32, 64, 128],bn=False).cuda()
            #lr = 0.001
        self.batch_size = 128

        cudnn.benchmark = True
        self.print("lamb1:{} lamb2:{} lr:{} nrate:{}".format(lamb1, lamb2, lr, noise_rate))
        self.op_type = op_type # "NAD"
        self.criterion = RSRLoss(lamb1, lamb2, self.model.A, noise_rate=noise_rate).cuda()
        self.print(op_type)
        if self.op_type == 'AD':
            self.optimizerA = optim.Adam(self.model.parameters(), lr=lr)
            self.optimizerB = optim.Adam([self.model.A], lr=lr*10)
            self.optimizerC = optim.Adam(self.model.parameters(), lr=lr * 10)
            #self.optimizerC = optim.Adam([{"params":self.model.encoder.parameters()}, {"params":self.model.A}],eps=1e-7,weight_decay=0.0005, lr=lr*10)
        else:
            # use adam always
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            #self.epochs = 250


    def train_step(self, x, y=None):
        x = torch.autograd.Variable(x.cuda())
        y, y_rsr, z, x_r = self.model(x)

        if self.op_type == 'AD':
            lossA = self.criterion.L21_error(x, x_r).mean() + self.criterion.proj_error() + self.criterion.pca_error(y, y_rsr).mean()
            # print(lossA)
            self.losses.update(lossA.item(), 1)
            # compute gradient and do SGD step
            self.optimizerA.zero_grad()
            lossA.backward()
            self.optimizerA.step()

            lossB = self.criterion.proj_error()
            self.optimizerB.zero_grad()
            lossB.backward()
            self.optimizerB.step()

            y, y_rsr, z, x_r = self.model(x)
            lossC = self.criterion.pca_error(y, y_rsr).mean()
            self.optimizerC.zero_grad()
            lossC.backward()
            self.optimizerC.step()
        else:
            loss = self.criterion(x, x_r, y, y_rsr)
            self.losses.update(loss.item(), 1)
            #print(loss.item())
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        cos_sim = []
        y_test = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.testloader):
                inputs = inputs.cuda()
                x = torch.autograd.Variable(inputs.view(inputs.shape[0], -1))
                inputs = torch.autograd.Variable(inputs)
                y, y_rsr, z, x_r = self.model(inputs)

                x_r = x_r.view(x_r.shape[0], -1)
                cs = F.cosine_similarity(x, x_r)
                cos_sim.append(cs)
                y_test.append(labels)

        cos_sim = torch.cat(cos_sim, dim=0).cpu().numpy()
        y_test = torch.cat(y_test, dim=0).cpu().numpy()
        return cos_sim, y_test


    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)
