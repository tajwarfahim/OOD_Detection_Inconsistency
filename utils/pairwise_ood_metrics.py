# Notes
# ------------------------------------------------------------------
# TPR_N = at N% False Positive Rate, what is the True Positive Rate?
# FPR_N = at N% True Positive Rate, what is the False Positive Rate?
# OOD examples = Positive, In-distribution examples = Negative

# OOD examples -> label 1, or are positive
# ID examples -> label 0, or are negative

# documentation help taken from:
# 1. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
# 3. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

# ID scores are higher in all calculation except AUROC and AUPR,
# we change the score accordinly by multiplying by -1.0 or 1.0

# For accumulator function across classes:
# 1. If function == "max": ID scores > OOD scores
# 2. If function == "min": OOD scores > ID scores
# 3. If function == "average": ID scores > OOD scores


# imports from packages
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

# from our scripts
from .other_baselines import fpr_and_fdr_at_recall


# OOD examples -> label 1
# ID examples -> label 0
def prepare_scores_and_labels(id_scores, ood_scores):
    assert isinstance(id_scores, np.ndarray)
    assert isinstance(ood_scores, np.ndarray)
    assert len(id_scores.shape) == 1
    assert len(ood_scores.shape) == 1

    combined_scores = np.append(id_scores, ood_scores, axis = 0)
    y = np.append(np.zeros_like(a = id_scores), np.ones_like(a = ood_scores), axis = 0)

    return combined_scores, y

def calculate_roc_auc(scores, y):
    return roc_auc_score(y_true = y, y_score = scores)

def draw_roc_curve(scores, y, save_path):
    baseline_scores = np.array([0 for i in range(y.shape[0])], dtype = np.float64)

    baseline_fpr, baseline_tpr, _ = roc_curve(y_true = y, y_score = baseline_scores)
    model_fpr, model_tpr, _ = roc_curve(y_true = y, y_score = scores)

    fig = plt.figure(figsize=(10.0, 7.0))
    plt.plot(baseline_fpr, baseline_tpr, 'b--')
    plt.plot(model_fpr, model_tpr, 'r-')

    plt.legend(["Baseline", "Model"], fontsize = 'x-large')
    plt.xlabel("False Positive Rate", fontsize = 'x-large')
    plt.ylabel("True Positive Rate", fontsize = 'x-large')
    plt.title("ROC Curve", fontsize = 'xx-large')

    if save_path is not None:
        plt.savefig(save_path)

    plt.close(fig = fig)

def calculate_pr_auc(scores, y):
    pr_auc = average_precision_score(y_true = y, y_score = scores)
    return pr_auc

def draw_pr_curve(scores, y, save_path):
    precision, recall, _ = precision_recall_curve(y_true = y, probas_pred = scores)

    # labels in y are 0 and 1, so we leverage this
    baseline = float(np.sum(y)) / y.shape[0]

    fig = plt.figure(figsize=(10.0, 7.0))
    plt.plot([0, 1], [baseline, baseline], 'b--')
    plt.plot(recall, precision, 'r-')

    plt.legend(["Baseline", "Model"], fontsize = 'x-large')
    plt.xlabel("Recall", fontsize = 'x-large')
    plt.ylabel("Precision", fontsize = 'x-large')
    plt.title("Precision Recall Curve", fontsize = 'xx-large')

    if save_path is not None:
        plt.savefig(save_path)

    plt.close(fig = fig)

class OODMetricsCalculator:
    def __init__(self, id_scores, ood_scores, id_dataset_name, ood_dataset_name):
        self.score_multiplier_map_AU = {"average": -1.0,
                                        "max": -1.0,
                                        "min": 1.0,
                                        "max - min": -1.0}

        self.id_scores = id_scores
        self.ood_scores = ood_scores

        self.combined_scores, self.y = prepare_scores_and_labels(id_scores = self.id_scores, ood_scores = self.ood_scores)

    def get_roc_auc_score(self, name_accumulator_across_classes):
        scores = self.combined_scores * self.score_multiplier_map_AU[name_accumulator_across_classes]
        auroc = calculate_roc_auc(scores = scores, y = self.y)
        auroc = round(auroc * 100, 1)

        return auroc

    def draw_model_roc_curve(self, name_accumulator_across_classes, save_path):
        scores = self.combined_scores * self.score_multiplier_map_AU[name_accumulator_across_classes]
        draw_roc_curve(scores = scores, y = self.y, save_path = save_path)

    def get_pr_auc_score(self, name_accumulator_across_classes, positive):
        scores = self.combined_scores * self.score_multiplier_map_AU[name_accumulator_across_classes]

        if positive == "in":
            scores = scores * (-1.0)
            y = 1 - self.y

        elif positive == "out":
            y = self.y

        else:
            raise ValueError("Given identification of positive dataset is not recognized.")

        y = np.array(y, dtype = np.int32)
        aupr = calculate_pr_auc(scores = scores, y = y)
        aupr = round(aupr * 100, 1)

        return aupr

    def draw_model_pr_curve(self, name_accumulator_across_classes, save_path):
        scores = self.combined_scores * self.score_multiplier_map_AU[name_accumulator_across_classes]
        draw_pr_curve(scores = scores, y = self.y, save_path = save_path)

    def get_fpr(self, name_accumulator_across_classes, recall_level):
        scores = self.combined_scores * self.score_multiplier_map_AU[name_accumulator_across_classes]
        fpr = fpr_and_fdr_at_recall(y_true = self.y, y_score = scores, recall_level = recall_level, pos_label = None)
        fpr = round(fpr * 100, 1)

        return fpr
