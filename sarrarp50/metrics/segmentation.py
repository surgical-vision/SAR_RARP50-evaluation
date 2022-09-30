import torch
import monai
import cv2
from pathlib import Path
import numpy as np
import warnings


def save_one_hot(root_dir, oh):
    # function to sotre a one hot tensor as separate images 
    for i, c in enumerate(oh):
        print(c.numpy().shape)
        cv2.imwrite(f'{root_dir}/{i}.png', c.numpy().astype(np.uint8)*255)
    exit()

def imread_one_hot(filepath, n_classes):
    # reads a segmentation mask stored as png and returns in in one-hot torch.tensor format
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError (filepath)
    if len(img.shape)==3:# if the segmentation mask was 3 channel, only keep the first
        img = img[...,0]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).requires_grad_(False)
    return monai.networks.utils.one_hot(img, n_classes, dim=1)



def get_val_func(metric, n_classes=9, fix_nans=False):
    # this function is intented for wrapping the meanIoU and meanNSD metric computation functions
    # It returns a error computation function that is able to parse reference 
    # and prediction segmentaiton samples in directory level. 
    def f(dir_pred, dir_ref):
        seg_ref_paths = sorted(list(dir_ref.iterdir()))
        dir_pred = Path(dir_pred)
        
        acc=[]
        with torch.no_grad():
            for seg_ref_p in seg_ref_paths:

                # load segmentation masks as one_hot torch tensors
                try:
                    ref = imread_one_hot(seg_ref_p, n_classes=n_classes+1)
                except FileNotFoundError:
                    raise
                try:
                    pred = imread_one_hot(dir_pred/seg_ref_p.name, n_classes=n_classes+1)
                except FileNotFoundError as e:
                    # if the prediciton file was not found, set all scores to zero and continue
                    acc.append([0]*n_classes) 
                    continue
                
                
                if fix_nans:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        err = metric(pred, ref)
                    # if both reference and predictions are zero, set the prediciton values to one
                    # this is required for NSD because otherise values are goint to
                    # be set to nan even though the prediciton is correct.
                    # in case either the pred or corresponding ref channel is zero
                    # NSD will resurn either 0 or nan and in those cases nan is 
                    # converted to zero
                    # find the zero channels in both ref and pred and create a mask 
                    # in the size of the final prediction.(1xn_channels)
                    r_m, p_m = ref.mean(axis=(2,3)), pred.mean(axis=(2,3))
                    mask = ((r_m ==0) * (p_m == 0))[:,1:]

                    # set the scores in cases where both ref and pred were full zero
                    #  to 1.
                    err[mask==True]=1
                    # in cases where either was full zero but the other wasn't 
                    # (score is nan ) set the corresponding score to 0
                    err.nan_to_num_()
                else:
                    err = metric(pred, ref)
                acc.append(err.detach().reshape(-1).tolist())
        return np.array(acc)# need to add axis and then multiply with the channel scales
    return f
    
def mIoU(dir_pred, dir_ref, n_classes=9):
    metric = monai.metrics.MeanIoU(include_background=False, reduction='mean', get_not_nans=False, ignore_empty=False)
    validation_func = get_val_func(metric, n_classes=n_classes)
    return validation_func(dir_pred/'segmentation', dir_ref/'segmentation')

def mNSD(dir_pred, dir_ref, n_classes = 9, channel_tau = [1]*9):
    metric = monai.metrics.SurfaceDiceMetric(channel_tau,
                                            include_background=False,
                                            reduction='mean')
    validation_func = get_val_func(metric, n_classes=n_classes, fix_nans=True)
    return validation_func(dir_pred/'segmentation', dir_ref/'segmentation')
    
