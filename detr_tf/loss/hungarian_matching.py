from typing import Union, Dict, Tuple
from itertools import product
import tensorflow as tf
import numpy as np

from .. import bbox
from scipy.optimize import linear_sum_assignment


def np_tf_linear_sum_assignment(matrix):

    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    #print(matrix.shape, target_indices, pred_indices)

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    #print('target_indices', target_indices)
    #print("pred_indices", pred_indices)

    return [target_indices, pred_indices, target_selector, pred_selector]


def hungarian_matching(
        t_bbox, t_class, p_bbox, p_class, fcost_class=1, fcost_bbox=5, fcost_giou=2, slice_preds=True) -> tuple:

    if slice_preds:
        size = tf.cast(t_bbox[0][0], tf.int32)
        t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
        t_class = tf.slice(t_class, [1, 0], [size, -1])
        t_class = tf.squeeze(t_class, axis=-1)

    # Convert frpm [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    p_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(t_bbox)

    softmax = tf.nn.softmax(p_class)

    # Classification cost for the Hungarian algorithom
    # On each prediction. We select the prob of the expected class
    cost_class = -tf.gather(softmax, t_class, axis=1)

    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = bbox.merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    iou, union = bbox.jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:, :, :2], _t_bbox_xy[:, :, :2])
    bottom_right = tf.math.maximum(_p_bbox_xy[:, :, 2:], _t_bbox_xy[:, :, 2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:, :, 0] * size[:, :, 1]
    cost_giou = -(iou - (area - union) / area)

    # Final hungarian cost matrix
    cost_matrix = fcost_bbox * cost_bbox + fcost_class * cost_class + fcost_giou * cost_giou

    selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool])
    target_indices = selectors[0]
    pred_indices = selectors[1]
    target_selector = selectors[2]
    pred_selector = selectors[3]

    return pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class
