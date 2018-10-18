# -*- coding: utf-8 -*-
import torch
from torch import nn
from .vgg16 import VGG16Extractor512, VGG16Extractor300


class SSD300(nn.Module):
    steps = (8, 16, 32, 64, 100, 300)
    # scale step=(0.9-0.2)/(5-1)=0.17
    # (30,60=300*0.2,300*0.37,...)
    # 'min_sizes': [30, 60, 111, 162, 213, 264]
    # 'max_sizes': [60, 111, 162, 213, 264, 315]
    box_size = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (38, 19, 10, 5, 3, 1)

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        self.extractor = VGG16Extractor300()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i],
                                          self.num_anchors[i] * 4,
                                          kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i],
                                          self.num_anchors[i] * self.num_classes,
                                          kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x).permute(0, 2, 3, 1).contiguous()
            cls_pred = self.cls_layers[i](x).permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


class SSD512(nn.Module):
    steps = (8, 16, 32, 64, 128, 256, 512)
    box_sizes = (35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (64, 32, 16, 8, 4, 2, 1)

    def __init__(self, num_classes):
        super(SSD512, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = (4, 6, 6, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256, 256)

        self.extractor = VGG16Extractor512()
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers += [nn.Conv2d(self.in_channels[i],
                                          self.num_anchors[i] * 4,
                                          kernel_size=3, padding=1)]
            self.cls_layers += [nn.Conv2d(self.in_channels[i],
                                          self.num_anchors[i] * self.num_classes,
                                          kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x).permute(0, 2, 3, 1).contiguous()
            cls_pred = self.cls_layers[i](x).permute(0, 2, 3, 1).contiguous()

            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


def test():
    net300 = SSD300(21)
    net512 = SSD512(21)
    print('***SSD300***')
    loc_preds, cls_preds = net300(torch.randn(1, 3, 300, 300))
    print(loc_preds.size(), cls_preds.size())
    print('------')
    print('***SSD512***')
    loc_preds, cls_preds = net512(torch.randn(1, 3, 512, 512))
    print(loc_preds.size(), cls_preds.size())
