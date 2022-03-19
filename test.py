from network import make_network
import tqdm
import torch
import time
import nms
from dataset.data_loader import make_data_loader
from train.model_utils.utils import load_network
from evaluator.make_evaluator import make_evaluator
import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument("config_file", help='/path/to/config_file.py')
parser.add_argument("--checkpoint", default='', help='/path/to/model_weight.pth')
parser.add_argument("--dataset", default='None', help='test dataset name')
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--eval", default='segm', help='evaluate the segmentation or detection result',
                    choices=['segm', 'bbox'])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--type", default='accuracy', help='evaluate the accuracy or speed',
                    choices=['speed', 'accuracy'])
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()

def get_cfg(args):
    cfg = importlib.import_module('configs.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.segm_or_bbox = args.eval
    cfg.test.test_stage = args.stage
    if args.dataset != 'None':
        cfg.test.dataset = args.dataset
    return cfg

def run_network(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()

    data_loader = make_data_loader(is_train=False, cfg=cfg)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(inp, batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader), '{} FPS'.format(len(data_loader) / total_time))

def run_evaluate(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()
    data_loader = make_data_loader(is_train=False, cfg=cfg)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp, batch)
        if cfg.test.with_nms:
            nms.post_process(output)
        evaluator.evaluate(output, batch)
    evaluator.summarize()

if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    if args.type == 'speed':
        run_network(cfg)
    else:
        run_evaluate(cfg)
