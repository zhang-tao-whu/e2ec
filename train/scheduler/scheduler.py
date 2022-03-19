from torch.optim.lr_scheduler import MultiStepLR
from collections import Counter


def make_lr_scheduler(optimizer, config):
    scheduler = MultiStepLR(optimizer, milestones=config.train.optimizer['milestones'],
                            gamma=config.train.optimizer['gamma'])
    return scheduler


def set_lr_scheduler(scheduler, config):
    scheduler.milestones = Counter(config.train.optimizer['milestones'])
    scheduler.gamma = config.train.optimizer['gamma']

