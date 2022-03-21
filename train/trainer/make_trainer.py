from .snake import NetworkWrapper
from .trainer import Trainer


def _wrapper_factory(network, cfg):
    return NetworkWrapper(network, with_dml=cfg.train.with_dml,
                          start_epoch=cfg.train.start_epoch, weight_dict=cfg.train.weight_dict)


def make_trainer(network, cfg):
    network = _wrapper_factory(network, cfg)
    return Trainer(network)
