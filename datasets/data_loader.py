# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ObjDetDataset(Dataset):
    def __init__(self, root, list_file, transform):
        '''
        :param root: the path where the images save
        :param list_file: decide which images to load, also contains
        the boxes as well its labels. "a.jpg xmin ymin xmax ymax label ..."
        :param transform: the transformation for img, box and label
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        if isinstance(list_file, list):
            temp_file = "/tmp/listfile.txt"
            os.system(f"cat {' '.join(list_file)} > {temp_file}")
            list_file = temp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            line_splitted = line.strip().split()
            self.fnames.append(line_splitted[0])
            num_boxes = (len(line_splitted) - 1) // 5
            boxes = []
            labels = []
            for i in range(num_boxes):
                xmin = line_splitted[1 + 5 * i]
                ymin = line_splitted[2 + 5 * i]
                xmax = line_splitted[3 + 5 * i]
                ymax = line_splitted[4 + 5 * i]
                c = line_splitted[5 * (i + 1)]
                boxes.append([float(xmin), float(ymin),
                              float(xmax), float(ymax)])
                labels.append(int(c))
            self.boxes.append(boxes)
            self.labels.append(labels)

    def __getitem__(self, idx):
        '''
        :return:
            ims: (tensor)
            boxes: (tensor)
            labels: (tensor)
        '''
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.FloatTensor(self.boxes[idx])
        labels = torch.LongTensor(self.labels[idx])
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        return img, boxes, labels

    def __len__(self):
        return self.num_imgs


if __name__ == '__main__':
    voc_dataset = ObjDetDataset(root="/home/yhuangcc/data/voc(07+12)/JPEGImages/",
                                list_file=["/home/yhuangcc/ObjectDetection/datasets/voc/voc07_trainval.txt",
                                           "/home/yhuangcc/ObjectDetection/datasets/voc/voc12_trainval.txt"],
                                transform=None)
    coco_dataset = ObjDetDataset(root="/home/yhuangcc/data/coco/images/train2017/",
                                 list_file="/home/yhuangcc/ObjectDetection/datasets/coco/coco17_train.txt",
                                 transform=None)
