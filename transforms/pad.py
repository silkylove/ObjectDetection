# -*- coding: utf-8 -*-
from PIL import Image


def pad(img, target_size):
    '''
    :param img: (PIL.Image)
    :param target_size: (tuple) (ow,oh)
    '''
    w, h = img.size
    canvas = Image.new('RGB', target_size)
    canvas.paste(img, (0, 0))
    return canvas
