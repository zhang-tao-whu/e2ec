import torch
from torch import nn


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class DMLoss(nn.Module):
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()

    def interpolation(self, poly, time=10):
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, 10)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), -1, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance
    
    def lossPred2NearestGt(self, ini_pred_poly, pred_poly, gt_poly):
        gt_poly_interpolation = self.interpolation(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_predto_nearestgt

    def lossGt2NearestPred(self, ini_pred_poly, pred_poly, gt_poly):
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, keyPointsMask):
        return self.setloss(ini_pred_poly, pred_polys_, gt_polys, keyPointsMask)