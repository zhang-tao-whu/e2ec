from importlib import import_module
from dataset.info import DatasetInfo


def _evaluator_factory(name, result_dir, anno_file, eval_format):
    file = import_module('evaluator.{}.snake'.format(name))
    if eval_format == 'segm':
        evaluator = file.Evaluator(result_dir, anno_file)
    else:
        evaluator = file.DetectionEvaluator(result_dir, anno_file)
    return evaluator


def make_evaluator(cfg):
    name = cfg.test.dataset.split('_')[0]
    anno_file = DatasetInfo.dataset_info[cfg.test.dataset]['anno_dir']
    eval_format = cfg.test.segm_or_bbox
    return _evaluator_factory(name, cfg.commen.result_dir, anno_file, eval_format)
