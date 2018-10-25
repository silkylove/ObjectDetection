# -*- coding: utf-8 -*-
import itertools
import math

import torch
import numpy as np
from utils.bbox import bbox_iou, cython_nms_o, change_bbox_order, bbox_nms


class Yolov3BboxCoder:
    def __init__(self, yolo_model):
        self.anchors = yolo_model.anchors
        self.num_classes = yolo_model.num_classes
        self.input_size = yolo_model.model_params['input_size']
        self.fm_size = [self.input_size / pow(2., i + 3) for i in range(3)]
        self.fm_size.reverse()
        self.steps = [self.input_size / fm_size for fm_size in self.fm_size]

    def encode(self, boxes, labels, iou_threshold=0.5):
        target_list = list()
        objmask_list = list()
        noobjmask_list = list()
        for i, ori_anchors in enumerate(self.anchors):
            in_h = in_w = int(self.fm_size[i])
            # self.input_size[0] / in_w, self.input_size[1] / in_h
            w_fm_stride, h_fm_stride = self.input_size / in_w, self.input_size / in_h
            anchors = [(a_w / w_fm_stride, a_h / h_fm_stride) for a_w, a_h in ori_anchors]
            num_anchors = len(anchors)
            obj_mask = torch.zeros(num_anchors, in_h, in_w)
            noobj_mask = torch.ones(num_anchors, in_h, in_w)
            tx = torch.zeros(num_anchors, in_h, in_w)
            ty = torch.zeros(num_anchors, in_h, in_w)
            tw = torch.zeros(num_anchors, in_h, in_w)
            th = torch.zeros(num_anchors, in_h, in_w)
            tconf = torch.zeros(num_anchors, in_h, in_w)
            tcls = torch.zeros(num_anchors, in_h, in_w, self.num_classes)

            for t in range(boxes.size(0)):
                # Convert to position relative to box
                gx = (boxes[t, 0].item() + boxes[t, 2].item()) / (2.0 * self.input_size) * in_w  # [0]
                gy = (boxes[t, 1].item() + boxes[t, 3].item()) / (2.0 * self.input_size) * in_h  # [1]
                gw = (boxes[t, 2].item() - boxes[t, 0].item()) / self.input_size * in_w  # [0]
                gh = (boxes[t, 3].item() - boxes[t, 1].item()) / self.input_size * in_h  # [1]
                if gw * gh == 0 or gx >= in_w or gy >= in_h:
                    continue

                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[anch_ious[0] > iou_threshold] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious, axis=1)

                # Masks
                obj_mask[best_n, gj, gi] = 1
                # Coordinates
                tx[best_n, gj, gi] = gx - gi
                ty[best_n, gj, gi] = gy - gj
                # Width and height
                tw[best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                # object
                tconf[best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[best_n, gj, gi, int(labels[t])] = 1

            obj_mask = obj_mask.view(-1, 1)
            noobj_mask = noobj_mask.view(-1, 1)
            tx = tx.view(-1, 1)
            ty = ty.view(-1, 1)
            tw = tw.view(-1, 1)
            th = th.view(-1, 1)
            tconf = tconf.view(-1, 1)
            tcls = tcls.view(-1, self.num_classes)
            target = torch.cat((tx, ty, tw, th, tconf, tcls), -1)
            target_list.append(target)
            objmask_list.append(obj_mask)
            noobjmask_list.append(noobj_mask)

        target = torch.cat(target_list, 0)
        obj_mask = torch.cat(objmask_list, 0)
        noobj_mask = torch.cat(noobjmask_list, 0)
        return target, torch.cat([obj_mask, noobj_mask], dim=1)

    def decode(self, batch_pred_bboxes, score_thresh=0.6, nms_thresh=0.45):

        box_corner = batch_pred_bboxes.new(batch_pred_bboxes.shape)
        box_corner[:, :, 0] = batch_pred_bboxes[:, :, 0] - batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 1] = batch_pred_bboxes[:, :, 1] - batch_pred_bboxes[:, :, 3] / 2
        box_corner[:, :, 2] = batch_pred_bboxes[:, :, 0] + batch_pred_bboxes[:, :, 2] / 2
        box_corner[:, :, 3] = batch_pred_bboxes[:, :, 1] + batch_pred_bboxes[:, :, 3] / 2

        # clip bounding box
        box_corner[:, :, 0::2] = box_corner[:, :, 0::2].clamp(min=0, max=1.0)
        box_corner[:, :, 1::2] = box_corner[:, :, 1::2].clamp(min=0, max=1.0)

        batch_pred_bboxes[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(batch_pred_bboxes))]
        for image_i, image_pred in enumerate(batch_pred_bboxes):
            # Filter out confidence scores below threshold
            conf_mask = (image_pred[:, 4] > score_thresh).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if image_pred.numel() == 0:
                continue

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 5:5 + self.num_classes], 1, keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            keep_index = bbox_nms(image_pred[:, :4],
                                  scores=image_pred[:, 4],
                                  nms_threshold=nms_thresh)

            output[image_i] = detections[keep_index]

        return output

    # def _get_default_boxes(self):
    #     boxes = []
    #     for i, fm_size in enumerate(self.fm_size):
    #         fm_size = int(fm_size)
    #         x_y_offset = meshgrid(fm_size, fm_size)
    #         x_y_offset = x_y_offset.repeat(len(self.anchors[i]), 1, 1).view(-1, 2)
    #         anchors = torch.FloatTensor(self.anchors[i]).view(3, 1, 2). \
    #             repeat(1, fm_size * fm_size, 1).contiguous().view(-1, 2)
    #         box = torch.cat([x_y_offset, anchors], dim=1)
    #         boxes.append(box / fm_size)
    #
    #     return torch.cat(boxes, dim=0)


def test():
    from .net import DarkNetYolov3
    model = DarkNetYolov3({"name": "darknet_53",
                           'pretrained': '/home/yhuangcc/ObjectDetection/models/yolov3/darknet53_weights_pytorch.pth',
                           'input_size': 416},
                          3).cuda()
    box_coder = Yolov3BboxCoder(model)
    boxes = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200], [200, 200, 300, 300]], dtype=torch.float).cuda()
    labels = torch.tensor([0, 1, 2], dtype=torch.long).cuda()
    targets, masks = box_coder.encode(boxes, labels)
    print(targets.size(), masks.size())
    output = box_coder.decode(torch.randn(1, 10647, 8).cuda(), 0.1)
    print(output[0].size())
    print(output)
