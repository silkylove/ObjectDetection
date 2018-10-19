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
from models.fpnssd import FPNSSD512, SSDBboxCoder

gpuid = 3

print('Loading model..')
net = FPNSSD512(num_classes=21).cuda(gpuid)
box_coder = SSDBboxCoder(net)

net = nn.DataParallel(net, [gpuid])
net.load_state_dict(torch.load('./checkpoint/ckpt.pt')['net'])
# net.load_state_dict(torch.load('./fpnssd512_20_trained.pth'))
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
    return img, boxes, labels


dataset = ObjDetDataset(root='/home/yhuangcc/data/voc(07+12)/JPEGImages/', \
                        list_file='/home/yhuangcc/ObjectDetection/datasets/voc/voc07_test.txt',
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

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
        for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
            if i % 500 == 0:
                print('%d/%d' % (i, len(dataloader)))
            gt_boxes.append(box_targets.squeeze(0))
            gt_labels.append(label_targets.squeeze(0))

            loc_preds, cls_preds = net(inputs.cuda(gpuid))
            box_preds, label_preds, score_preds = box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=0.01)

            pred_boxes.append(box_preds)
            pred_labels.append(label_preds)
            pred_scores.append(score_preds)

    return voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True)


# the eval time should be around 700s
print('Start to eval...')
start = time.time()
result = eval(net, dataloader)
pickle.dump(result, open('./voc_eval_result.pk', 'wb'))
print('ap', result['ap'])
print('map', result['map'])
print(f'eval ends... consuming time {time.time()-start:.4f}s')
