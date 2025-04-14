
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc
def AP(anomal_scores, labels):
    precision, recall, th=precision_recall_curve(np.squeeze(labels, axis=0), np.squeeze(anomal_scores))
    frame_ap = auc(recall,precision)
    return frame_ap
def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha[0]*list1[i]+alpha[1]*list2[i]))   
    return list_result
def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    # mu = np.mean(support)
    filter = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter

def filt(input, dim=9, range=302, mu=21, normalized=False):
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    filter_2d = gaussian_filter(np.arange(1, range), mu)

    frame_scores = input  # convolve(input, filter_3d)
    # frame_scores = frame_scores.max((1, 2))

    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    frame_scores = np.nan_to_num(frame_scores, nan=0.)
    if normalized:
        frame_scores = normalize_scores(frame_scores)
    return frame_scores
def normalize_scores(pred):
    return (pred - np.min(pred)) / (np.max(pred) - np.min(pred))