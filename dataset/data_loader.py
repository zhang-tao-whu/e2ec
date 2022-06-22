import torch
import torch.utils.data
from .collate_batch import collate_batch
from .info import DatasetInfo

def make_dataset(dataset_name, is_test, cfg):
    info = DatasetInfo.dataset_info[dataset_name]
    if is_test:
        from .test import coco, cityscapes, cityscapesCoco, sbd, kitti
        dataset_dict = {'coco': coco.CocoTestDataset, 'cityscapes': cityscapes.Dataset,
                        'cityscapesCoco': cityscapesCoco.CityscapesCocoTestDataset,
                        'kitti': kitti.KittiTestDataset, 'sbd': sbd.SbdTestDataset}
        dataset = dataset_dict[info['name']]
    else:
        from .train import coco, cityscapes, cityscapesCoco, sbd, kitti
        dataset_dict = {'coco': coco.CocoDataset, 'cityscapes': cityscapes.Dataset,
                        'cityscapesCoco': cityscapesCoco.CityscapesCocoDataset,
                        'kitti': kitti.KittiDataset, 'sbd': sbd.SbdDataset}
        dataset = dataset_dict[info['name']]
    dataset = dataset(info['anno_dir'], info['image_dir'], info['split'], cfg)
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_ddp_data_sampler(dataset, shuffle):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return sampler

def make_batch_data_sampler(sampler, batch_size, drop_last):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    return batch_sampler

def make_train_loader(cfg):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    dataset = make_dataset(dataset_name, is_test=False, cfg=cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = cfg.train.num_workers
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader

def make_test_loader(cfg, is_distributed=True):
    batch_size = 1
    shuffle = True if is_distributed else False
    drop_last = False
    dataset_name = cfg.test.dataset

    dataset = make_dataset(dataset_name, is_test=True, cfg=cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader


def make_data_loader(is_train=True, is_distributed=False, cfg=None):
    if is_train:
        return make_train_loader(cfg), make_test_loader(cfg, is_distributed)
    else:
        return make_test_loader(cfg, is_distributed)

def make_demo_loader(data_root=None, cfg=None):
    from .demo_dataset import Dataset
    batch_size = 1
    shuffle = False
    drop_last = False
    dataset = Dataset(data_root, cfg)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(sampler, batch_size, drop_last)
    num_workers = 1
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader

def make_ddp_train_loader(cfg):
    batch_size = cfg.train.batch_size
    shuffle = True
    drop_last = False
    dataset_name = cfg.train.dataset

    dataset = make_dataset(dataset_name, is_test=False, cfg=cfg)
    sampler = make_ddp_data_sampler(dataset, shuffle)
    collator = collate_batch
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=batch_size,
        collate_fn=collator,
        pin_memory=False,
        drop_last=drop_last
    )
    return data_loader

def make_ddp_data_loader(is_train=True, is_distributed=False, cfg=None):
    if is_train:
        return make_ddp_train_loader(cfg), make_test_loader(cfg, is_distributed)
    else:
        return make_test_loader(cfg, is_distributed)

