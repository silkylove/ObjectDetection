# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()
        self.features = vgg16(pretrained=True)
        self.norm4 = L2Norm(512, 20)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)

        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                                   nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                                   nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3),
                                    nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        xs = []
        for i in range(23):
            x = self.features.features[i](x)
        xs.append(self.norm4(x))

        for i in range(23, 30):
            x = self.features.features[i](x)
        x = self.pooling(x)

        x = self.conv7(self.conv6(x))
        xs.append(x)

        for i in range(8, 12):
            x = eval(f'self.conv{i}(x)')
            xs.append(x)

        return xs


class VGG16Extractor512(nn.Module):
    def __init__(self):
        super(VGG16Extractor512, self).__init__()
        self.features = vgg16(pretrained=True)
        self.norm4 = L2Norm(512, 20)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)

        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                                   nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                                   nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.conv11 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))

        self.conv12 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=4, padding=1))

    def forward(self, x):
        xs = []
        for i in range(23):
            x = self.features.features[i](x)
        xs.append(self.norm4(x))

        for i in range(23, 30):
            x = self.features.features[i](x)
        x = self.pooling(x)

        x = self.conv7(self.conv6(x))
        xs.append(x)

        for i in range(8, 13):
            x = eval(f'self.conv{i}(x)')
            xs.append(x)

        return xs


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class L2Norm(nn.Module):
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameter(scale)

    def reset_parameter(self, scale):
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    ## silghtly different in the last M
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
    return model


def test():
    ex300 = VGG16Extractor300()
    ex512 = VGG16Extractor512()
    print('***VGG16_300***')
    for p in ex300(torch.randn(1, 3, 300, 300)):
        print(p.size())
    print('------')
    print('***VGG16_512***')
    for p in ex512(torch.randn(1, 3, 512, 512)):
        print(p.size())
