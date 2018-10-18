# -*- coding: utf-8 -*-
import torch
from torch import nn
from .fpn import FPN50


class RetinaNet(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        fms = self.fpn(x)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, 4)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, dim=1), torch.cat(cls_preds, dim=1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


def test():
    net = RetinaNet(21)
    print('***RetainNet***')
    loc_preds, cls_preds = net(torch.randn(1, 3, 640, 640))
    print(loc_preds.size(), cls_preds.size())
