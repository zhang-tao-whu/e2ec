from network import make_network
from train.trainer.make_trainer import make_trainer
from train.optimizer.optimizer import make_optimizer
from train.scheduler.scheduler import make_lr_scheduler
from train.recorder.recorder import make_recorder
from dataset.data_loader import make_ddp_data_loader
from train.model_utils.utils import load_model, save_model, load_network
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np

def setup(world_size):
    dist.init_process_group("nccl", init_method='env://', world_size=world_size)

def cleanup():
    dist.destroy_process_group()

parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default='coco')
parser.add_argument("--checkpoint", default="None")
parser.add_argument("--type", default="continue")
parser.add_argument("--bs", default="2")
parser.add_argument("--dml", default="True")
parser.add_argument("--gpus", default="2")

parser.add_argument("--local_rank", type=int)

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file)
    if args.bs != 'None':
        cfg.train.batch_size = int(args.bs)
    if args.dml != 'True':
        cfg.train.with_dml = False
    return cfg
cfg = get_cfg(args)

def train(network, cfg, rank):
    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = network.to(rank)
    trainer = make_trainer(network, cfg)
    trainer.network = DDP(trainer.network, device_ids=[rank], find_unused_parameters=True)
    network = trainer.network.module.net
    optimizer = make_optimizer(network, cfg)
    scheduler = make_lr_scheduler(optimizer, cfg)
    recorder = make_recorder(cfg.commen.record_dir)
    evaluator = make_evaluator(cfg)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    if args.type == 'finetune':
        begin_epoch = load_network(network, model_dir=args.checkpoint, map_location=map_location)
    else:
        begin_epoch = load_model(network, optimizer, scheduler, recorder, args.checkpoint, map_location=map_location)

    train_loader, val_loader = make_ddp_data_loader(cfg=cfg)

    for epoch in range(begin_epoch, cfg.train.epoch):
        train_loader.sampler.set_epoch(epoch)
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()
        if rank == 0:
            if (epoch + 1) % cfg.train.save_ep == 0:
                save_model(network, optimizer, scheduler, recorder, epoch,
                       cfg.commen.model_dir)
            if (epoch + 1) % cfg.train.eval_ep == 0:
                trainer.val(epoch, val_loader, evaluator, recorder)
    return network

def run(rank):
    setup(int(args.gpus))
    seed = rank + 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    network = make_network.get_network(cfg)
    train(network, cfg, rank)
    cleanup()
    return


if __name__ == "__main__":
    rank = args.local_rank
    torch.cuda.set_device(rank)
    run(rank)
