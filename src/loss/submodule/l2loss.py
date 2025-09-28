


import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        # self.t_valid = 0.0001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        # mask = (gt > self.t_valid).type_as(pred).detach()

        # d = torch.pow(pred - gt, 2) * mask
        d = torch.pow(pred - gt, 2)

        d = torch.sum(d, dim=[1, 2, 3])
        # num_valid = torch.sum(mask, dim=[1, 2, 3])

        # loss = d / (num_valid + 1e-8)
        total_num = torch.numel(gt)
        assert total_num != 0
        loss = d / (total_num + 1e-8)
        loss = loss.sum()

        return loss
