# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .backbone import backbone_fn


class DarkNetYolov3(nn.Module):
    anchors = [[[116, 90], [156, 198], [373, 326]],
               [[30, 61], [62, 45], [59, 119]],
               [[10, 13], [16, 30], [33, 23]]]

    def __init__(self, model_params, num_classes):
        '''
        :param model_params:
            {
                "name": "darknet_53",
                "pretrained": "./darknet53_weights_pytorch.pth",
            }
        :param num_classes: int
        '''
        super(DarkNetYolov3, self).__init__()
        self.num_classes = num_classes
        self.model_params = model_params
        #  backbone
        _backbone_fn = backbone_fn[self.model_params["name"]]
        self.backbone = _backbone_fn(self.model_params["pretrained"])
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        final_out_filter0 = len(self.anchors[0]) * (5 + self.num_classes)
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(self.anchors[1]) * (5 + self.num_classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(self.anchors[2]) * (5 + self.num_classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

    def forward(self, x, infer=False):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        # out0 = out0.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 5 + self.num_classes)
        #
        # out1 = out1.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 5 + self.num_classes)
        #
        # out2 = out2.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 5 + self.num_classes)
        #
        # out = torch.cat([out0, out1, out2], dim=1)
        #
        # out[:, :, :2] = torch.sigmoid(out[:, :, :2])
        # out[:, :, 4:] = torch.sigmoid(out[:, :, 4:])
        layer_out_list = [out0, out1, out2]
        detect_list = list()
        prediction_list = list()
        for i in range(len(layer_out_list)):
            batch_size, _, grid_size_h, grid_size_w = layer_out_list[i].size()
            feat_stride = self.model_params['input_size'] // layer_out_list[i].size(-1)
            in_anchors = self.anchors[i]
            bbox_attrs = 4 + 1 + self.num_classes
            num_anchors = len(in_anchors)

            anchors = [(a[0] / feat_stride, a[1] / feat_stride) for a in in_anchors]

            layer_out = layer_out_list[i].view(batch_size, num_anchors * bbox_attrs, grid_size_h * grid_size_w)
            layer_out = layer_out.contiguous().view(batch_size, num_anchors, bbox_attrs, grid_size_h * grid_size_w)
            layer_out = layer_out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, bbox_attrs)

            # Sigmoid the  centre_X, centre_Y. and object confidencce
            layer_out[:, :, 0] = torch.sigmoid(layer_out[:, :, 0])
            layer_out[:, :, 1] = torch.sigmoid(layer_out[:, :, 1])
            layer_out[:, :, 4] = torch.sigmoid(layer_out[:, :, 4])

            # Softmax the class scores
            layer_out[:, :, 5: 5 + self.num_classes] = torch.sigmoid((layer_out[:, :, 5: 5 + self.num_classes]))

            prediction_list.append(layer_out)
            if infer:
                detect_out = layer_out.clone()
                # Add the center offsets
                grid_len_h = np.arange(grid_size_h)
                grid_len_w = np.arange(grid_size_w)
                a, b = np.meshgrid(grid_len_w, grid_len_h)

                x_offset = torch.FloatTensor(a).view(-1, 1)
                y_offset = torch.FloatTensor(b).view(-1, 1)

                x_offset = x_offset.to(detect_out.device)
                y_offset = y_offset.to(detect_out.device)

                x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(1, -1, 2)
                x_y_offset = x_y_offset.repeat(num_anchors, 1, 1).view(-1, 2).unsqueeze(0)

                detect_out[:, :, :2] += x_y_offset

                # log space transform height and the width
                anchors = torch.FloatTensor(anchors)

                anchors = anchors.to(detect_out.device)

                anchors = anchors.contiguous().view(3, 1, 2) \
                    .repeat(1, grid_size_h * grid_size_w, 1).contiguous().view(-1, 2).unsqueeze(0)
                detect_out[:, :, 2:4] = torch.exp(detect_out[:, :, 2:4]) * anchors

                detect_out[:, :, 0] /= grid_size_w
                detect_out[:, :, 1] /= grid_size_h
                detect_out[:, :, 2] /= grid_size_w
                detect_out[:, :, 3] /= grid_size_h

                detect_list.append(detect_out)

        return torch.cat(prediction_list, 1) if not infer else torch.cat(detect_list, 1)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m


def test():
    model_params = {"name": "darknet_53",
                    'pretrained': '/home/yhuangcc/ObjectDetection/models/yolov3/darknet53_weights_pytorch.pth',
                    'input_size': 416}
    m = DarkNetYolov3(model_params, 20).cuda()
    x = torch.randn(1, 3, 416, 416).cuda()
    y1, y2, y3 = m(x)
    print(y1[0].size())
    print(y1[1].size())
    print(y1[2].size())
    print(y2.size())
    print(y3.size())
    print(y3)
