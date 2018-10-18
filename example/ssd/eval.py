# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transforms import resize
from datasets import ObjDetDataset
from evaluations.voc_eval import voc_eval
from models.ssd import SSD512, SSDBboxCoder

from PIL import Image

print('Loading model..')
net = SSD512(num_classes=21).cuda(3)
net = nn.DataParallel(net, [3])
net.load_state_dict(torch.load('./checkpoint/ckpt.pt')['net'])
net.eval()

print('Preparing dataset..')
img_size = 512


def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size, img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels


box_coder = SSDBboxCoder(net)

dataset = ObjDetDataset(root='/home/yhuangcc/data/voc(07+12)/JPEGImages/', \
                        list_file='/home/yhuangcc/ObjectDetection/datasets/voc/voc07_test.txt',
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

# box_coder.anchor_boxes = box_coder.anchor_boxes.cuda(3)
pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('/home/yhuangcc/ObjectDetection/datasets/voc/voc07_test_difficult.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(np.array(d))


def eval(net, dataloader):
    with torch.no_grad():
        idx = 0
        for i, (inputs, _, _) in enumerate(dataloader):
            print('%d/%d' % (i, len(dataloader)))

            loc_preds, cls_preds = net(inputs.cuda(3))
            for j in range(inputs.size(0)):
                box_preds, label_preds, score_preds = box_coder.decode(
                    loc_preds[j].cpu().data.squeeze(),
                    F.softmax(cls_preds[j].cpu().data.squeeze(), dim=1),
                    score_threshold=0.01)
                pred_boxes.append(box_preds)
                pred_labels.append(label_preds)
                pred_scores.append(score_preds)

            for j in range(idx, idx + inputs.size(0)):
                gt_boxes.append(torch.FloatTensor(dataset.boxes[j]))
                gt_labels.append(torch.LongTensor(dataset.labels[j]))

            idx += inputs.size(0)
    return voc_eval(pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels, gt_difficults,
                    iou_thresh=0.5, use_07_metric=True)


print('Start to eval...')
start = time.time()
result = eval(net, dataloader)
pickle.dump(result, open('./voc_eval_result.pk', 'wb'))
print('ap', result['ap'])
print('map', result['map'])
print(f'eval ends... consuming time {time.time()-start:.4f}s')
