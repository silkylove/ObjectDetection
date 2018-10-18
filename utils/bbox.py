# -*- coding: utf-8 -*-
import torch
import numpy as np


#TODO: change to be cython with numpy
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


def bbox_nms(boxes, scores, threshold=0.5):
    '''
    Non maximum suppression
    :param boxes: (tensor) [N,4]
    :param scores: (tensor) [N,]
    :param threshold: (float)
    :return:
        keep: (tensor) selected indices
    '''
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0]
        keep.append(i)
        overlap = bbox_iou(boxes[order[1:]], boxes[i]).squeeze()
        ids = (overlap < threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


