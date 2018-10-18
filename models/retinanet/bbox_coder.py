# -*- coding: utf-8 -*-
import math
import torch
import itertools

from utils import meshgrid
from utils.bbox import bbox_iou, bbox_nms, change_bbox_order


# TODO: change to be cython with numpy
class RetinaBboxCoder:
    def __init__(self):
        self.anchor_areas = (32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.)
        self.aspect_ratios = (1 / 2., 1 / 1., 2 / 1.)
        self.scale_ratios = (1., pow(2, 1 / 3.), pow(2, 2 / 3.))
        self.anchor_boxes = self._get_anchor_boxes(input_size=torch.FloatTensor([640., 640.]))

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.
        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj,4].
          labels: (tensor) object class labels, sized [#obj,].
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''

        anchor_boxes = self.anchor_boxes
        ious = bbox_iou(anchor_boxes, boxes)
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        boxes = change_bbox_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_bbox_order(anchor_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[max_ids]
        # cls_targets[max_ious<0.5] = 0
        # ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        # cls_targets[ignore] = -1                  # mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_threshold=0.5, nms_threshold=0.5):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [#anchors,4].
          cls_preds: (tensor) predicted conf, sized [#anchors,#classes].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        anchor_boxes = self.anchor_boxes  # xywh

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)  # [#anchors,]
        ids = score > score_threshold
        ids = ids.nonzero().squeeze()  # [#obj,]
        keep = bbox_nms(boxes[ids], score[ids], threshold=nms_threshold)
        return boxes[ids][keep], labels[ids][keep], score[ids][keep]

    def _get_anchor_wh(self):
        '''
        Compute anchor width and height for each feature map
        :return:
            anchor_wh: (tensor) anchor wh, [#fm, #anchors_per_cell, 2]
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:
                h = math.sqrt(s / ar)
                w = h * ar
                for sr in self.scale_ratios:
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.FloatTensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''
        Compute anchor boxes for each feature map
        :param input_size: (tensor) model input size of (w,h)
        :return:
            anchor_boxes: (tensor) [#anchors, 4]
            #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        anchor_wh = self._get_anchor_wh()
        # need to be careful
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)
            wh = anchor_wh[i].view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy - wh / 2., xy + wh / 2.], dim=3)
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes)


def test():
    box_coder = RetinaBboxCoder()
    print(box_coder.anchor_boxes.size())
    boxes = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200], [200, 200, 300, 300]], dtype=torch.float)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    loc_targets, cls_targets = box_coder.encode(boxes, labels)
    print(loc_targets.size(), cls_targets.size())
    de_boxes, de_labels, de_scores = box_coder.decode(loc_targets, torch.randn(76725, 3))
    print(de_boxes)
