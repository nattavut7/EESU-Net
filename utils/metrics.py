import numpy as np


def dice_score(pred, target, cls):
    pred_c = pred == cls
    target_c = target == cls
    inter = np.logical_and(pred_c, target_c).sum()
    denom = pred_c.sum() + target_c.sum()
    return (2 * inter) / (denom + 1e-8)


def iou_score(pred, target, cls):
    pred_c = pred == cls
    target_c = target == cls
    inter = np.logical_and(pred_c, target_c).sum()
    union = np.logical_or(pred_c, target_c).sum()
    return inter / (union + 1e-8)


def sensitivity(pred, target, cls):
    pred_c = pred == cls
    target_c = target == cls
    tp = np.logical_and(pred_c, target_c).sum()
    fn = np.logical_and(~pred_c, target_c).sum()
    return tp / (tp + fn + 1e-8)


def specificity(pred, target, cls):
    pred_c = pred == cls
    target_c = target == cls
    tn = np.logical_and(~pred_c, ~target_c).sum()
    fp = np.logical_and(pred_c, ~target_c).sum()
    return tn / (tn + fp + 1e-8)


def precision(pred, target, cls):
    pred_c = pred == cls
    target_c = target == cls
    tp = np.logical_and(pred_c, target_c).sum()
    fp = np.logical_and(pred_c, ~target_c).sum()
    return tp / (tp + fp + 1e-8)
