# -*- coding: utf-8 -*-
import torch
from torch import nn


class YOLOv3Loss(nn.Module):
    def __init__(self):
        super(YOLOv3Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, prediction, targets, masks):
        objmask = masks[..., 0]
        noobjmask = masks[..., 1]
        # Get outputs
        x = prediction[..., 0]  # Center x
        y = prediction[..., 1]  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = prediction[..., 4]  # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        # Get targets
        tx = targets[..., 0]  # Center x
        ty = targets[..., 1]  # Center y
        tw = targets[..., 2]  # Width
        th = targets[..., 3]  # Height
        tcls = targets[..., 5:]  # Cls pred.

        #  losses.
        loss_x = self.bce_loss(x[objmask == 1], tx[objmask == 1]) / objmask.sum()
        loss_y = self.bce_loss(y[objmask == 1], ty[objmask == 1]) / objmask.sum()
        loss_w = self.mse_loss(w[objmask == 1], tw[objmask == 1]) / objmask.sum()
        loss_h = self.mse_loss(h[objmask == 1], th[objmask == 1]) / objmask.sum()
        loss_obj = self.bce_loss(conf[objmask == 1], objmask[objmask == 1]) / objmask.sum()
        loss_noobj = self.bce_loss(conf[noobjmask == 1], objmask[noobjmask == 1]) / noobjmask.sum()
        loss_cls = self.bce_loss(pred_cls[objmask == 1], tcls[objmask == 1]) / objmask.sum()

        loss_loc = loss_x + loss_y + loss_w + loss_h
        #  total loss = losses * weight
        loss = loss_loc * 2.5 + loss_obj + loss_noobj * 0.5 + loss_cls
        return loss_loc, loss_cls, loss
