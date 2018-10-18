# -*- coding: utf-8 -*-
import torch
import random
from PIL import Image


def resize(img, boxes, size, max_size=1000, random_interpolation=False):
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    method = random.choice([Image.BOX, Image.NEAREST, Image.HAMMING,
                            Image.BICUBIC, Image.LANCZOS, Image.BILINEAR]) \
        if random_interpolation else Image.BILINEAR
    img = img.resize((ow, oh), method)
    if boxes is not None:
        boxes = boxes * torch.FloatTensor([sw, sh, sw, sh])
    return img, boxes
