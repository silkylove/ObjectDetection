# -*- coding: utf-8 -*-

import math
import torch
import itertools

from utils.bbox import bbox_nms, bbox_iou, change_bbox_order, cython_nms_o


class SSDBboxCoder:
    def __init__(self, ssd_model):
        self.steps = ssd_model.steps
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.fm_sizes = ssd_model.fm_sizes
        self.variances = (0.1, 0.2)
        self.default_boxes = self._get_default_boxes()

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

        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1].item()
            return (i[j], j)

        default_boxes = self.default_boxes
        default_boxes = change_bbox_order(default_boxes, 'xywh2xyxy')

        ious = bbox_iou(default_boxes, boxes)
        index = torch.empty(len(default_boxes), dtype=torch.long).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] > 0.5)
        if mask.any():
            index[mask] = ious[mask].max(1)[1]

        boxes = boxes[index.clamp(min=0)]
        boxes = change_bbox_order(boxes, 'xyxy2xywh')
        default_boxes = change_bbox_order(default_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:] / self.variances[0]
        loc_wh = torch.log(boxes[:, 2:] / default_boxes[:, 2:]) / self.variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], dim=1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        xy = loc_preds[:, :2] * self.variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * self.variances[1]) * self.default_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)
        box_preds = box_preds.clamp(min=0, max=max(self.steps) * min(self.fm_sizes))

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes - 1):
            score = cls_preds[:, i + 1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask]
            score = score[mask]

            keep = bbox_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.empty_like(keep).fill_(i))
            scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)
        return boxes, labels, scores

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                x = (w + 0.5) * self.steps[i]
                y = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((x, y, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i + 1])
                boxes.append((x, y, s, s))
                for ar in self.aspect_ratios[i]:
                    boxes.append((x, y, s * math.sqrt(ar), s * math.sqrt(ar)))
                    boxes.append((x, y, s / math.sqrt(ar), s / math.sqrt(ar)))

        return torch.FloatTensor(boxes)


def test():
    from .net import FPNSSD512
    box_coder = SSDBboxCoder(FPNSSD512(21))
    print(box_coder.default_boxes.size())
    boxes = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200], [200, 200, 300, 300]], dtype=torch.float)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    loc_targets, cls_targets = box_coder.encode(boxes, labels)
    print(loc_targets.size(), cls_targets.size())
    de_boxes, de_labels, de_scores = box_coder.decode(loc_targets, torch.randn(24564, 3))
    print(de_boxes)
