# -*- coding: utf-8 -*-
import torch
import numpy as np


# TODO: change to be cython with numpy
def change_bbox_order(boxes, order):
    '''
    Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    :param boxes: (tensor) [N,4]
    :param order: (str) 'xyxy2xywh' or 'xywh2xyxy'
    '''

    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a, b = boxes[:, :2], boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a + b) / 2, b - a], 1)
    return torch.cat([a - b / 2, a + b / 2], 1)


def bbox_clamp(boxes, xmin, ymin, xmax, ymax):
    boxes[:, 0].clamp_(xmin, xmax)
    boxes[:, 1].clamp_(ymin, ymax)
    boxes[:, 2].clamp_(xmin, xmax)
    boxes[:, 3].clamp_(ymin, ymax)
    return boxes


def bbox_select(boxes, xmin, ymin, xmax, ymax):
    '''
    :return:
        (tensor) selected boxes, [M,4]
        (tensor) selected mask, [N,]
    '''
    mask = (boxes[:, 0] >= xmin) & (boxes[:, 1] > ymin) \
           & (boxes[:, 2] < xmax) & (boxes[:, 3] < ymax)
    boxes = boxes[mask, :]
    return boxes, mask


def bbox_iou(box1, box2):
    '''
    Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    :param box1: (tensor) [N,4]
    :param box2: (tensor) [M,4]
    :return:
        iou: (tensor) [N,M]
    '''
    if len(box1.size()) == 1:
        box1.unsqueeze_(0)
    if len(box2.size()) == 1:
        box2.unsqueeze_(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# very slow
def bbox_nms_torch(boxes, scores, threshold=0.5):
    '''
    Non maximum suppression
    :param boxes: (tensor) [N,4]
    :param scores: (tensor) [N,]
    :param threshold: (float)
    :return:
        keep: (tensor) selected indices
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (overlap < threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def bbox_nms(boxes, scores, threshold=0.5):
    '''
    Non maximum suppression
    :param boxes: (tensor) [N,4]
    :param scores: (tensor) [N,]
    :param threshold: (float)
    :return:
        keep: (tensor) selected indices
    '''
    is_torch = False
    if not isinstance(boxes, np.ndarray):
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        is_torch = True
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        if len(order) == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0]
        keep.append(i)

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        ids = np.where(overlap < threshold)[0]
        if len(ids) == 0:
            break
        order = order[ids + 1]
    return keep if not is_torch else torch.LongTensor(keep)


import extensions.nms.src.cython_nms as cython_nms


def cython_nms_o(bboxes, scores=None, nms_threshold=0.5):
    """Apply classic DPM-style greedy NMS."""
    is_torch = False
    if not isinstance(bboxes, np.ndarray):
        bboxes = bboxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        is_torch = True

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1, 1)

    dets = np.concatenate((bboxes, scores), 1)
    if dets.shape[0] == 0:
        return []

    keep = cython_nms.nms(dets, nms_threshold)
    return keep if not is_torch else torch.tensor(keep).long()
