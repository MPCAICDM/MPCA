from helpers.base_helper import TrainTestHelper, transform_train, transform_test
from utils import  get_channels_axis,save_roc_pr_curve_data, show_roc_pr_curve_data
from models.encoders_decoders import CAE_pytorch
from keras2pytorch_dataset import trainset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest


class CAEHelper(TrainTestHelper):
    def __init__(self, n_channels, *args, **kwargs):
        super(CAEHelper, self).__init__(*args, **kwargs)
        self.method_tag = "cae"

        self.n_channels = n_channels
        self.model = CAE_pytorch(in_channels=self.n_channels).cuda()
        self.batch_size = 128

        cudnn.benchmark = True
        self.criterion = nn.MSELoss()
        # use adam always
        self.optimizer = optim.Adam(self.model.parameters(), eps=1e-7, weight_decay=0.0005)
        #self.epochs = 250


    def train_step(self, x, y=None):
        inputs = torch.autograd.Variable(x.cuda())
        outputs = self.model(inputs)
        loss = self.criterion(inputs, outputs)

        self.losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_scores(self):
        self.model.eval()
        losses = []
        reps = []
        y_test = []
        for batch_idx, (inputs, labels) in enumerate(self.testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            rep = self.model.encode(inputs)
            outputs = self.model.decode(rep)
            loss = outputs.sub(inputs).pow(2).view(outputs.size(0), -1)
            loss = loss.sum(dim=1, keepdim=False)
            losses.append(loss.data.cpu())
            reps.append(rep.data.cpu())
            y_test.append(labels.data.cpu())
        losses = torch.cat(losses, dim=0)
        reps = torch.cat(reps, dim=0)
        y_test = torch.cat(y_test, dim=0)
        losses = losses.numpy()
        losses = losses - losses.min()
        losses = losses / (1e-8 + losses.max())
        scores = 1 - losses
        return scores, reps.numpy(), y_test.numpy()

    def test(self, is_show=True):
        scores, reps, y_test = self.compute_scores()
        if is_show:
            roc_auc, pr_auc_norm, pr_auc_anom = show_roc_pr_curve_data(scores, y_test)
            self.print("auroc:{}, pr_auc_norm:{}, pr_auc_anom:{}".format(roc_auc, pr_auc_norm, pr_auc_anom), False)
        else:
            res_file_path = self.get_result_file_path()
            save_roc_pr_curve_data(scores, y_test, res_file_path)

            # Use reps to train iforest
            clf = IsolationForest(contamination=self.p, n_jobs=4).fit(reps)
            scores_iforest = clf.decision_function(reps)
            iforest_file_path = self.get_result_file_path('iforest')
            save_roc_pr_curve_data(scores_iforest, y_test, iforest_file_path)