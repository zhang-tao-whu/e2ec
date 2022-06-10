import torch.nn as nn
from .backbone.dla import DLASeg
from .detector_decode.refine_decode import Decode
from .evolve.evolve import Evolution
import torch

class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.test_stage = cfg.test.test_stage

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv, use_dcn=cfg.model.use_dcn)
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                    coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                    min_ct_score=cfg.test.ct_score)
        self.gcn = Evolution(evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.commen.down_ratio)

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        if 'test' not in batch['meta']:
            self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)
        output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)
        return output

def get_network(cfg):
    network = Network(cfg)
    return network
