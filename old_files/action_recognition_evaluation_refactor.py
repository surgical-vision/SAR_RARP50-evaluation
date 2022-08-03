import numpy as np



def split_sequence(discrete_labels):
    # takes as inputs 
    discrete_labels = discrete_labels[:,1]
    # find the samples where the action label changes
    idxs = [0] + list(np.nonzero(np.diff(discrete_labels))[0] + 1)
    # find what is the label in every change
    labels = np.array([discrete_labels[idx] for idx in idxs])
    # append the list and find all the action segments. you do that by spliting the
    # sequense in places where the action label changes
    idxs.append(len(discrete_labels))
    intervals = np.array([(idxs[i], idxs[i + 1]) for i in range(len(idxs)-1)])
    return intervals, labels


def filter_background(intervals, labels, bg_class):
    intervals = intervals[labels!=bg_class]
    labels = labels[labels!=bg_class]
    return intervals, labels
    


def accuracy(P, Y):
    return np.mean(P==Y)


def f1k(reference, predictions, n_classes=0, bg_class=None, k=10):
    # def overlap_(p,y, n_classes, bg_class, overlap):

    # true_intervals = np.array(segment_intervals(y))
    # true_labels = segment_labels(y)
    # pred_intervals = np.array(segment_intervals(p))
    # pred_labels = segment_labels(p)
    
    
    intervals_ref, labels_ref = split_sequence(reference)
    intervals_pred, labels_pred = split_sequence(predictions)

    # Remove background labels
    if bg_class is not None:
        intervals_ref, labels_ref = filter_background(intervals_ref, labels_ref, bg_class)
        intervals_pred, labels_pred = filter_background(intervals_pred, labels_pred, bg_class)

    n_true = labels_ref.shape[0]
    n_pred = labels_pred.shape[0]

    # We keep track of the per-class TP, and FP.
    # In the end we just sum over them though.
    TP = np.zeros(n_classes, np.float)
    FP = np.zeros(n_classes, np.float)
    true_used = np.zeros(n_true, np.float)

    for j in range(n_pred):
        # Compute IoU against all others
        intersection = np.minimum(intervals_pred[j,1], intervals_ref[:,1]) - np.maximum(intervals_pred[j,0], intervals_ref[:,0])
        union = np.maximum(intervals_pred[j,1], intervals_ref[:,1]) - np.minimum(intervals_pred[j,0], intervals_ref[:,0])
        IoU = (intersection / union)*(labels_pred[j]==labels_ref)

        # Get the best scoring segment
        idx = IoU.argmax()

        # If the IoU is high enough and the true segment isn't already used
        # Then it is a true positive. Otherwise is it a false positive.
        if IoU[idx] >= k/100 and not true_used[idx]:
            TP[labels_pred[j]] += 1
            true_used[idx] = 1
        else:
            FP[labels_pred[j]] += 1

    TP = TP.sum()
    FP = FP.sum()
    # False negatives are any unused true segment (i.e. "miss")
    FN = n_true - true_used.sum()

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2 * (precision*recall) / (precision+recall)

    # If the prec+recall=0, it is a NaN. Set these to 0.
    F1 = np.nan_to_num(F1)

    return F1

