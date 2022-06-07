import torch.nn as nn
from .utils import FocalLoss, DMLoss, sigmoid
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net, with_dml=True, start_epoch=10, weight_dict=None):
        super(NetworkWrapper, self).__init__()
        self.with_dml = with_dml
        self.net = net
        self.ct_crit = FocalLoss()
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.weight_dict = weight_dict
        self.start_epoch = start_epoch
        if with_dml:
            self.dml_crit = DMLoss(type='smooth_l1')
        else:
            self.dml_crit = self.py_crit

    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        if 'test' in batch['meta']:
            return output
        epoch = batch['epoch']
        scalar_stats = {}
        loss = 0.

        keyPointsMask = batch['keypoints_mask'][batch['ct_01']]
        
        ct_loss = self.ct_crit(sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        num_polys = len(output['poly_init'])
        if num_polys == 0:
            init_py_loss = torch.sum(output['poly_init']) * 0.
            coarse_py_loss = torch.sum(output['poly_coarse']) * 0.
        else:
            init_py_loss = self.py_crit(output['poly_init'], output['img_gt_polys'])
            coarse_py_loss = self.py_crit(output['poly_coarse'], output['img_gt_polys'])
        scalar_stats.update({'init_py_loss': init_py_loss})
        scalar_stats.update({'coarse_py_loss': coarse_py_loss})
        loss += init_py_loss * self.weight_dict['init']
        loss += coarse_py_loss * self.weight_dict['coarse']

        py_loss = 0
        n = len(output['py_pred']) - 1 if self.with_dml else len(output['py_pred'])
        for i in range(n):
            if num_polys == 0:
                part_py_loss = torch.sum(output['py_pred'][i]) * 0.0
            else:
                part_py_loss = self.py_crit(output['py_pred'][i], output['img_gt_polys'])
            py_loss += part_py_loss / len(output['py_pred'])
            scalar_stats.update({'py_loss_{}'.format(i): part_py_loss})
        loss += py_loss * self.weight_dict['evolve']

        if self.with_dml and epoch >= self.start_epoch and num_polys != 0:
            dm_loss = self.dml_crit(output['py_pred'][-2],
                                    output['py_pred'][-1],
                                    output['img_gt_polys'],
                                    keyPointsMask)
            scalar_stats.update({'end_set_loss': dm_loss})
            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']
        else:
            dm_loss = torch.sum(output['py_pred'][-1]) * 0.0
            scalar_stats.update({'end_set_loss': dm_loss})
            loss += dm_loss / len(output['py_pred']) * self.weight_dict['evolve']

        scalar_stats.update({'loss': loss})

        return output, loss, scalar_stats

