# -*- coding: utf-8 -*-
import os
import math
import time
import logging
import random
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.fpnssd import FPNSSD512, SSDBboxCoder
from loss import SSDLoss
from datasets import ObjDetDataset
from transforms import resize, random_flip, random_paste, random_distort, random_crop
from utils import AverageMeter


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


logger = logging.getLogger('InfoLog')


class Trainer:
    def __init__(self, config):
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.img_size = 512
        self.epochs = config.epochs
        self.start_epoch = 0

        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)


        self.datasets = {'train': ObjDetDataset(config.img_root, config.train_list, self.transform(train=True)),
                         'test': ObjDetDataset(config.img_root, config.test_list, self.transform(train=False))}

        self.loaders = {'train': DataLoader(self.datasets['train'], batch_size=self.batch_size,
                                            shuffle=True, pin_memory=True, num_workers=8),
                        'test': DataLoader(self.datasets['test'], batch_size=self.batch_size,
                                           shuffle=False, pin_memory=True, num_workers=8)}

        self.idx2label = dict(enumerate(np.loadtxt(config.label_file, np.object).tolist()))
        self.num_classes = 1 + len(self.idx2label)
        self.net = FPNSSD512(self.num_classes)
        self.box_coder = SSDBboxCoder(self.net)
        self.net = self.net.cuda()

        if config.distributed:
            self.net = nn.DataParallel(self.net)

        self.criterion = SSDLoss(self.num_classes)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.99))
        # with adam, the lr should be started from 1e-4
        self.lr_decay = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_decay,
                                                       gamma=config.lr_decay_rate)

        self.best_loss = float('inf')

        if config.resume:
            logger.info('***Resume from checkpoint***')
            state = torch.load(os.path.join(self.ckpt_dir, 'ckpt.pt'))
            self.net.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer.load_state_dict(state['optim'])
            self.lr_decay.load_state_dict(state['lr_decay'])
            self.lr_decay.last_epoch = self.start_epoch - 1

    def train_and_test(self):
        self.start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            self.lr_decay.step()
            logger.info(f"Epoch :{epoch}")
            self.train()
            logger.info(f"Test starts...")
            test_loss = self.test()
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.save({'net': self.net.state_dict(),
                           'best_loss': test_loss,
                           'epoch': epoch,
                           'optim': self.optimizer.state_dict(),
                           'lr_decay': self.lr_decay.state_dict()})

    def train(self):
        losses = AverageMeter()
        self.net.train()
        for i, (imgs, loc_targets, cls_targets) in enumerate(self.loaders['train']):
            imgs = imgs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            loc_preds, cls_preds = self.net(imgs)
            loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss = loc_loss + cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item())
            if i % 100 == 0:
                logger.info(f"Train: [{i}/{len(self.loaders['train'])}] | "
                            f"Time: {timeSince(self.start_time)} | "
                            f"loc_loss: {loc_loss.item():.4f} | "
                            f"cls_loss:{cls_loss.item():.4f} | "
                            f"Loss: {losses.avg:.4f}")

    def test(self):
        losses = AverageMeter()
        self.net.eval()
        with torch.no_grad():
            for i, (imgs, loc_targets, cls_targets) in enumerate(self.loaders['test']):
                imgs = imgs.cuda()
                loc_targets = loc_targets.cuda()
                cls_targets = cls_targets.cuda()

                loc_preds, cls_preds = self.net(imgs)
                loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                loss = loc_loss + cls_loss

                losses.update(loss.item(), imgs.size()[0])

            logger.info(f"Test: [{i}/{len(self.loaders['test'])}] | "
                        f"Time: {timeSince(self.start_time)} | "
                        f"loc_loss: {loc_loss.item():.4f} | "
                        f"cls_loss:{cls_loss.item():.4f} | "
                        f"Loss: {losses.avg:.4f}")
        return losses.avg

    def save(self, state):
        torch.save(state, os.path.join(self.ckpt_dir, 'ckpt.pt'))
        logger.info('***Saving model***')

    def transform(self, train=True):
        if train:
            def transform_train(img, boxes, labels):
                img = random_distort(img)
                if random.random() < 0.5:
                    img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123, 116, 103))
                img, boxes, labels = random_crop(img, boxes, labels)
                img, boxes = resize(img, boxes, size=(self.img_size, self.img_size), random_interpolation=False)
                img, boxes = random_flip(img, boxes)
                img = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])(img)
                boxes, labels = self.box_coder.encode(boxes, labels)
                return img, boxes, labels

            return transform_train
        else:
            def transform_test(img, boxes, labels):
                img, boxes = resize(img, boxes, size=(self.img_size, self.img_size))

                img = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])(img)
                boxes, labels = self.box_coder.encode(boxes, labels)
                return img, boxes, labels

            return transform_test
