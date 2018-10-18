# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def vis_image_bbox(img, boxes=None, label_names=None, scores=None):
    '''
    :param img: (PIL.Image/tensor)
    :param boxes: (tensor/array)
    :param label_names: (list)
    :param scores: (list/array)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)
    ax.imshow(img)

    if boxes is not None:
        for i, bb in enumerate(boxes):
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0] + 1
            height = bb[3] - bb[1] + 1
            ax.add_patch(plt.Rectangle(xy, width, height,
                                       fill=False,
                                       edgecolor='red',
                                       linewidth=2))
            caption = []
            if label_names is not None:
                caption.append(f"{label_names[i]}")

            if scores is not None:
                caption.append(f"{scores[i]:.2f}")

            if len(caption) > 0:
                ax.text(bb[0], bb[1],
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
    plt.show()



