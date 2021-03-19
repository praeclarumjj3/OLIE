import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable
import collections

def visualize_maps(maps, name):
    x = maps.cpu() 
    dim = int(x.shape[0])
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5)+1,int(dim**0.5)+1,figsize=(16,16))
    for j in range(dim):
        r = int(j/int((dim**0.5)+1))
        c = int(j%int((dim**0.5)+1))
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(name))

def visualize_single_map(mapi, name):
    x = mapi.cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, (ax) = plt.subplots(1,1)
    ax.imshow(x)
    ax.set_title("Map")
    ax.axis('off')
    f.savefig('visualizations/{}.jpg'.format(name))

def point_nms(heat, kernel=2):
    # kernel must be 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay / soft nms
    delay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'linear':
        delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
        delay_coefficient, _ = delay_matrix.min(0)
    else:
        delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = cate_scores * delay_coefficient

    return cate_scores_update


def mask_nms(cate_labels, seg_masks, sum_masks, cate_scores, nms_thr=0.5):
    n_samples = len(cate_scores)
    if n_samples == 0:
        return []

    keep = seg_masks.new_ones(cate_scores.shape)
    seg_masks = seg_masks.float()

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            label_j = cate_labels[j]
            if label_i != label_j:
                continue
            # overlaps
            inter = (mask_i * mask_j).sum()
            union = sum_masks[i] + sum_masks[j] - inter
            if union > 0:
                if inter / union > nms_thr:
                    keep[j] = False
            else:
                keep[j] = False
    return keep

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def visualize_instance_map(img, pred, name):
    colors = loadmat('datasets/color150.mat')['colors']
    
    img = img.cpu() 
    img = img.permute(1, 2, 0).numpy()
    img = img.astype(np.uint8)
    
    pred = as_numpy(pred.squeeze(0).cpu())
    pred = np.int32(pred)
    
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    Image.fromarray(im_vis).save(
        os.path.join('visualizations', name+'.jpg'))
