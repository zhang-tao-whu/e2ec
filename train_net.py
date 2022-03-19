from network import make_network
from train.trainer.make_trainer import make_trainer
from train.optimizer.optimizer import make_optimizer
from train.scheduler.scheduler import make_lr_scheduler
from train.recorder.recorder import make_recorder
from dataset.data_loader import make_data_loader
from train.model_utils.utils import load_model, save_model, load_network
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib
import torch

parser = argparse.ArgumentParser()

parser.add_argument("config_file")
parser.add_argument("--checkpoint", default="None")
parser.add_argument("--type", default="continue")
parser.add_argument("--bs", default="None")
parser.add_argument("--dml", default="True")
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file)
    if args.bs != 'None':
        cfg.train.batch_size = int(args.bs)
    if args.dml != 'True':
        cfg.train.with_dml = False
    return cfg

def train(network, cfg):
    trainer = make_trainer(network, cfg)
    optimizer = make_optimizer(network, cfg)
    scheduler = make_lr_scheduler(optimizer, cfg)
    recorder = make_recorder(cfg.commen.record_dir)
    evaluator = make_evaluator(cfg)

    if args.type == 'finetune':
        begin_epoch = load_network(network, model_dir=args.checkpoint)
    else:
        begin_epoch = load_model(network, optimizer, scheduler, recorder, args.checkpoint)

    train_loader, val_loader = make_data_loader(cfg=cfg)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.train.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch,
                       cfg.commen.model_dir)

        if (epoch + 1) % cfg.train.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network

def main():
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    network = make_network.get_network(cfg)
    train(network, cfg)

if __name__ == "__main__":
    main()
