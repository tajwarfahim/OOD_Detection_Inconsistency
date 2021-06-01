# imports from packages
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import scipy

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

# imports from our scripts
from .other_baselines import get_ood_scores_odin
from .other_baselines import get_Mahalanobis_score
from .other_baselines import sample_estimator
from .other_baselines import fpr_and_fdr_at_recall

def get_msp_score(model, dataloader):
    model.eval()

    all_scores = []
    with torch.no_grad():
        for img, _ in dataloader:
            if torch.cuda.is_available():
                img = img.cuda()

            scores = torch.nn.functional.softmax(model.forward(img), dim = 1).detach().cpu().numpy()
            max_scores = np.max(a = scores, axis = 1, keepdims = False)

            all_scores.append(max_scores)


    msp_scores = np.concatenate(all_scores, axis = 0)

    assert len(msp_scores.shape) == 1
    assert msp_scores.shape[0] == len(dataloader.dataset)

    return msp_scores

def get_accuracy(model, dataloader):
    model.eval()
    num_correct = 0

    with torch.no_grad():
        for img, label in dataloader:
            if torch.cuda.is_available():
                img = img.cuda()

            scores = torch.nn.functional.softmax(model.forward(img), dim = 1).detach().cpu().numpy()
            pred = np.argmax(scores, axis = 1)
            targets = label.detach().cpu().numpy()

            assert pred.shape == targets.shape
            assert len(pred.shape) == 1

            right_indices = pred == targets
            corrects = scores[right_indices].shape[0]
            num_correct = num_correct + corrects


    print("Num correct: ", num_correct)
    accuracy = float(num_correct) / len(dataloader.dataset)
    return accuracy


class MSPScorer:
    def __init__(self, model, id_train_dataloader, id_test_dataloader, ood_test_dataloader, num_classes, id_train_dataset_mean):
        self.model = model
        self.id_train_dataloader = id_train_dataloader
        self.id_test_dataloader = id_test_dataloader
        self.ood_test_dataloader = ood_test_dataloader
        self.num_classes = num_classes
        self.id_train_dataset_mean = id_train_dataset_mean

        print("Mean: ", self.id_train_dataset_mean)

    def calculate_msp_scores(self):
        self.id_msp_scores = -1.0 * get_msp_score(model = self.model, dataloader = self.id_test_dataloader)
        self.ood_msp_scores = -1.0 * get_msp_score(model = self.model, dataloader = self.ood_test_dataloader)

    def calculate_odin_scores(self, temperature, noise):
        ood_num_examples = len(self.ood_test_dataloader.dataset)

        self.id_odin_score, _, _ = get_ood_scores_odin(loader = self.id_test_dataloader,
                                                       net = self.model,
                                                       bs = self.id_test_dataloader.batch_size,
                                                       ood_num_examples = ood_num_examples,
                                                       T = temperature,
                                                       noise = noise,
                                                       in_dist = True,
                                                       mean = self.id_train_dataset_mean)

        self.ood_odin_score = get_ood_scores_odin(loader = self.ood_test_dataloader,
                                                  net = self.model,
                                                  bs = self.ood_test_dataloader.batch_size,
                                                  ood_num_examples = ood_num_examples,
                                                  T = temperature,
                                                  noise = noise,
                                                  in_dist = False,
                                                  mean = self.id_train_dataset_mean)

    def calculate_mahalanobis_scores(self, noise):
        ood_num_examples = len(self.ood_test_dataloader.dataset)
        batch_size = self.ood_test_dataloader.batch_size
        num_batches = int(ood_num_examples / batch_size)

        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x)
        if torch.cuda.is_available():
            temp_x = temp_x.cuda()

        temp_list = self.model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0

        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        sample_mean, precision = sample_estimator(model = self.model,
                                                  num_classes = self.num_classes,
                                                  feature_list = feature_list,
                                                  train_loader = self.id_train_dataloader)

        self.id_mahalanobis_score = get_Mahalanobis_score(model = self.model,
                                                          test_loader = self.id_test_dataloader,
                                                          num_classes = self.num_classes,
                                                          sample_mean = sample_mean,
                                                          precision = precision,
                                                          layer_index = count - 1,
                                                          magnitude = noise,
                                                          num_batches = num_batches,
                                                          in_dist = True,
                                                          mean = self.id_train_dataset_mean)

        self.ood_mahalanobis_score = get_Mahalanobis_score(model = self.model,
                                                           test_loader = self.ood_test_dataloader,
                                                           num_classes = self.num_classes,
                                                           sample_mean = sample_mean,
                                                           precision = precision,
                                                           layer_index = count - 1,
                                                           magnitude = noise,
                                                           num_batches = num_batches,
                                                           in_dist = False,
                                                           mean = self.id_train_dataset_mean)


    def choose_score(self, score_type):
        if score_type == "msp":
            scores = np.append(self.ood_msp_scores, self.id_msp_scores, axis = 0)
        elif score_type == "odin":
            scores = np.append(self.ood_odin_score, self.id_odin_score, axis = 0)
        elif score_type == "mahalanobis":
            scores = np.append(self.ood_mahalanobis_score, self.id_mahalanobis_score, axis = 0)
        else:
            raise ValueError("Given score type is not recognized.")

        return scores


    def calculate_auroc(self, score_type):
        y = [1  for i in range(len(self.ood_test_dataloader.dataset))] + [0 for i in range(len(self.id_test_dataloader.dataset))]
        y = np.array(y, dtype = np.int32)
        scores = self.choose_score(score_type = score_type)

        auroc = roc_auc_score(y_true = y, y_score = scores)
        auroc = round(auroc * 100, 1)

        return auroc

    def calculate_aupr(self, score_type, positive):
        scores = self.choose_score(score_type = score_type)

        if positive == "in":
            y = [0  for i in range(len(self.ood_test_dataloader.dataset))] + [1 for i in range(len(self.id_test_dataloader.dataset))]
            scores = (-1.0) * scores
        elif positive == "out":
            y = [1  for i in range(len(self.ood_test_dataloader.dataset))] + [0 for i in range(len(self.id_test_dataloader.dataset))]
        else:
            raise ValueError("Given value of positive dataset not recognized.")

        y = np.array(y, dtype = np.int32)
        aupr = average_precision_score(y_true = y, y_score = scores)
        aupr = round(aupr * 100, 1)

        return aupr

    def calculate_accuracy(self):
        accuracy = get_accuracy(model = self.model, dataloader = self.id_test_dataloader)
        accuracy = round(accuracy * 100, 2)

        return accuracy

    def calculate_fpr(self, score_type, recall_level):
        scores = self.choose_score(score_type = score_type)
        y = [1  for i in range(len(self.ood_test_dataloader.dataset))] + [0 for i in range(len(self.id_test_dataloader.dataset))]
        y = np.array(y, dtype = np.int32)

        fpr = fpr_and_fdr_at_recall(y_true = y, y_score = scores, recall_level = recall_level, pos_label = None)
        fpr = round(fpr * 100, 1)

        return fpr
