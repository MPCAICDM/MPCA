from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data, show_avg_scores
from models.ae_backbone import CAE_backbone, AE_backbone
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AERSEHelper(TrainTestHelper):
    def __init__(self, n_channels, h, w, lr=0.00025, flatten_size=128, *args, **kwargs):
        super(AERSEHelper, self).__init__(*args, **kwargs)
        self.method_tag = "ae_backbone"

        self.n_channels = n_channels
        if self.dataset_name in ['mnist', 'caltech101', 'fashion-mnist-rsrae', 'fashion-mnist',
                                 'tinyimagenet', 'cifar10']:
            self.model = CAE_backbone(input_shape=(n_channels, h, w),
                                       hidden_layer_sizes=[32, 64, 128], bn=False,
                                       flatten_size=flatten_size).cuda()
        else:
            self.model = AE_backbone(input_shape=n_channels,
                                             hidden_layer_sizes=[512, 256, 128], bn=False,
                                             flatten_size=flatten_size).cuda()

        cudnn.benchmark = True
        self.criterion = nn.MSELoss(size_average=False)
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-6, lr=lr)


    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        x_r = self.model(inputs)
        loss = self.criterion(inputs, x_r)
        self.losses.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        losses = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            x = inputs.cuda()
            with torch.no_grad():
                x_r = self.model(x)
                loss = x_r.sub(x).pow(2).view(x_r.size(0), -1)
                loss = loss.sum(dim=1, keepdim=False)
                losses.append(loss.data.cpu())
                y_test.append(labels.data.cpu())
        losses = torch.cat(losses, dim=0)
        y_test = torch.cat(y_test, dim=0)
        losses = losses.numpy()
        #losses = losses - losses.min()
        #losses = losses / (1e-8 + losses.max())
        scores = -losses
        return scores, y_test.numpy()

    def test(self, is_show=True):
        scores, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
            mean_pos, mean_neg, std_pos, std_neg = show_avg_scores(scores, y_test)
            self.print("mean_pos:{}, mean_neg:{} std_pos:{} std_neg:{}".format(mean_pos, mean_neg, std_pos, std_neg))
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

    def save(self):
        scores, y_test = self.compute_scores()
        res_file_path = self.get_result_file_path()
        print(res_file_path)
        save_roc_pr_curve_data(scores, y_test, res_file_path)