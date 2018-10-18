# -*- coding: utf-8 -*-
import sys

sys.path.append('..')
import random
import matplotlib.pyplot as plt
from transforms import random_crop, random_paste, resize, random_flip
from visualizations import vis_image_bbox
from datasets.data_loader import ObjDetDataset
from torchvision import transforms


def train_transform(img, boxes, labels):
    img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=600, random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    print(img.size)
    img = transforms.ToTensor()(img)
    print(boxes)
    return img, boxes, labels


voc_dataset = ObjDetDataset(root="/home/yhuangcc/data/voc(07+12)/JPEGImages/",
                            list_file=["/home/yhuangcc/ObjectDetection/datasets/voc/voc07_trainval.txt",
                                       "/home/yhuangcc/ObjectDetection/datasets/voc/voc12_trainval.txt"],
                            transform=train_transform)
coco_dataset = ObjDetDataset(root="/home/yhuangcc/data/coco/images/train2017/",
                             list_file="/home/yhuangcc/ObjectDetection/datasets/coco/coco17_train.txt",
                             transform=train_transform)

r1 = random.randint(0, len(voc_dataset))
r2 = random.randint(0, len(coco_dataset))
print(voc_dataset.fnames[r1])
vis_image_bbox(*voc_dataset[r1])
plt.close()
plt.pause(0.1)
print(coco_dataset.fnames[r2])
vis_image_bbox(*coco_dataset[r2])
plt.close()