import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_map(results, gt_labels):
    "Evaluate mAP of a dataset"
    results = np.asarray(results)
    gt_labels = np.asarray(gt_labels)
    assert results.shape[0] == gt_labels.shape[0]
    # try:
    APs = average_precision_score(gt_labels, results, average=None)
    # except ValueError:
    # print(gt_labels)
    # print(results)
    mAP = np.nanmean(APs)
    return mAP, APs


def eval_auroc(results, gt_labels):
    "Evaluate AUROC of a dataset"
    results = np.asarray(results)
    gt_labels = np.asarray(gt_labels)
    assert results.shape[0] == gt_labels.shape[0]
    AUROCs = roc_auc_score(gt_labels, results, average=None)
    AUROC = np.nanmean(AUROCs)
    return AUROC, AUROCs


def eval_map_subset(results, gt_labels, subset):
    results_subset = np.take(results, indices=subset, axis=1)
    gt_labels_subset = np.take(gt_labels, indices=subset, axis=1)
    assert results_subset.shape[0] == gt_labels_subset.shape[0]
    APs = average_precision_score(gt_labels_subset, results_subset, average=None)
    mAP = np.nanmean(APs)

    return mAP


def evaluate_slc(predict_p, gt_labels):
    # Calculate top-1 accuracy and error rate
    top1 = 0
    for i in range(len(predict_p)):
        highest_prob = np.argmax(predict_p[i])
        if gt_labels[i][highest_prob] == 1:
            top1 += 1

    top1 /= len(predict_p)
    top1error = 1 - top1

    return top1, top1error


def evaluate_mlc(
    predict_p,
    gt_labels,
    head_classes,
    middle_classes,
    tail_classes,
    calculate_auroc=True,
):
    mAP, APs = eval_map(predict_p, gt_labels)
    if head_classes is not None:
        mAP_head = eval_map_subset(predict_p, gt_labels, head_classes)
    else:
        mAP_head = None
    if middle_classes is not None:
        mAP_middle = eval_map_subset(predict_p, gt_labels, middle_classes)
    else:
        mAP_middle = None
    if tail_classes is not None:
        mAP_tail = eval_map_subset(predict_p, gt_labels, tail_classes)
    else:
        mAP_tail = None

    if calculate_auroc:
        AUROC, AUROCs = eval_auroc(predict_p, gt_labels)
    else:
        AUROC, AUROCs = None, None

    return mAP, APs, mAP_head, mAP_middle, mAP_tail, AUROC, AUROCs


def evaluate_mlc_no_split(predict_p, gt_labels):
    mAP, APs = eval_map(predict_p, gt_labels)
    AUROC, AUROCs = eval_auroc(predict_p, gt_labels)

    return mAP, APs, AUROC, AUROCs


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class BCEWithLogitsLoss(nn.Module):
    def __init__(
        self, label_smoothing=0.0, reduction="mean", weight=None, pos_weight=None
    ):
        super(BCEWithLogitsLoss, self).__init__()
        assert (
            0 <= label_smoothing < 1
        ), "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(
            reduction=reduction, weight=weight, pos_weight=pos_weight
        )

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = (
                target * positive_smoothed_labels
                + (1 - target) * negative_smoothed_labels
            )

        loss = self.bce_with_logits(input, target)
        return loss
