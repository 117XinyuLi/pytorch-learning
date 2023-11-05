import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold


def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
                                                        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
                                      labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))# nrof_pairs: number of pairs
    nrof_thresholds = len(thresholds)# nrof_thresholds: number of thresholds
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))# tpr: true positive rate
    fprs = np.zeros((nrof_folds, nrof_thresholds))# fpr: false positive rate
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold using the training set
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)

        # Calculate evaluation metrics on the test set
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])

        # Calculate accuracy on the test set using the best threshold
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set],
                                                      labels[test_set])
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, thresholds[best_threshold_index]# tpr&fpr will be used to draw ROC curve(the change of threshold causes the change of tpr&fpr


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)# np.less: return the boolean array of whether dist<threshold
    tp = np.sum(np.logical_and(predict_issame, actual_issame))# tp: true positive
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))# fp: false positive: predict as same but actually not
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))# tn: true negative
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))# fn: false negative: predict as not same but actually same

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)# tpr = tp/(tp+fn) = tp/actual same
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)# fpr = fp/(fp+tn) = fp/actual not same
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    # given the far target, calculate the validation rate

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)# val: validation rate(true positive rate)
    far = np.zeros(nrof_folds)# far: false accept rate(false positive rate)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            # f: interpolate the thresholds according to the far_train
            threshold = f(far_target)# threshold: the threshold that gives FAR = far_target
        else:
            threshold = 0.0

        # Calculate evaluation metrics on the test set using the threshold
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))# tp
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))# fp
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)# val = tpr = tp/actual same
    far = float(false_accept) / float(n_diff)# far = fpr = fp/actual not same
    return val, far
