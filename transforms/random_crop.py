# -*- coding: utf-8 -*-
import math
import random

import torch

from utils.bbox import bbox_iou, bbox_clamp


def random_crop(img, boxes, labels, min_scale=0.3, max_aspect_ratio=2.):
    '''
    :param img: (PIL.Image)
    :param boxes: (tensor) [N,4]
    :param labels: (tensor) [N,]
    :param min_scale: (float) minimal image width/height scale
    :param max_aspect_ratio: (float) maximum width/height aspect ratio
    :return:
        img, boxes, labels
    '''
    imw, imh = img.size
    params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
    ## ????
    for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(100):
            scale = random.uniform(min_scale, 1)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            w = int(imw * scale * math.sqrt(aspect_ratio))
            h = int(imh * scale / math.sqrt(aspect_ratio))

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)

            roi = torch.FloatTensor([[x, y, x + w, y + h]])
            ious = bbox_iou(boxes, roi)
            if ious.min() >= min_iou:
                params.append((x, y, w, h))
                break

    x, y, w, h = random.choice(params)
    img = img.crop((x, y, x + w, y + h))

    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    mask = (center[:, 0] >= x) & (center[:, 0] <= x + w) \
           & (center[:, 1] >= y) & (center[:, 1] <= y + h)
    if mask.any():
        boxes = boxes[mask] - torch.FloatTensor([x, y, x, y])
        boxes = bbox_clamp(boxes, 0, 0, w, h)
        labels = labels[mask]
    else:
        boxes = torch.FloatTensor([[0, 0, 0, 0]])
        labels = torch.LongTensor([0])
    return img, boxes, labels
