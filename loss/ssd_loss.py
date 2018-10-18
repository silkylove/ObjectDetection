# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class SSDLoss(nn.Module):
    def __init__(self, num_classes):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0
        batch_size = pos.size()[0]
        num_pos = pos.sum().item()

        mask = pos.unsqueeze(2).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction='sum')

        cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes),
                                   cls_targets.view(-1, ), reduction='none')
        cls_loss = cls_loss.view(batch_size, -1)

        # ??
        cls_loss[cls_targets < 0] = 0

        neg = self._head_negative_mining(cls_loss, pos)
        cls_loss = cls_loss[pos | neg].sum()

        return loc_loss / num_pos, cls_loss / num_pos

    def _head_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].
        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)
        _, idx = cls_loss.sort(1)
        _, rank = idx.sort(1)
        num_neg = 3 * pos.sum(1)
        neg = rank < num_neg[:, None]
        return neg
