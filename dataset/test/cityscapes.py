import cv2
import os
import numpy as np
import glob
import torch.utils.data as data
import json
from ..train.cityscapes import JSON_DICT
from ..train.utils import augment

class Dataset(data.Dataset):
    def __init__(self, anno_file, data_root, split, cfg):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.split = split
        self.anns = np.array(self.read_dataset(anno_file)[:])
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = JSON_DICT

    def read_dataset(self, ann_files):
        if not isinstance(ann_files, tuple):
            ann_files = (ann_files, )
        ann_file = []
        for ann_file_dir in ann_files:
            ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))
        return ann_file

    def process_info(self, fname, data_root):
        with open(fname, 'r') as f:
            ann = json.load(f)

        fname = fname.split('/')[-1]
        city = fname.split('_')[0]
        image_id = fname[:-5]
        info = [city, image_id + '_leftImg8bit.png']
        img_path = os.path.join(data_root, self.split, '/'.join(info))
        img_id = image_id
        return fname, img_path, img_id

    def __getitem__(self, index):
        data_input = {}

        ann = self.anns[index]
        image_name, image_path, image_id = self.process_info(ann, self.data_root)
        img = cv2.imread(image_path)

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )

        data_input.update({'inp': inp})
        #meta = {'center': center, 'scale': scale, 'test': '', 'img_id': image_id, 'ann': ann}
        meta = {'center': center, 'img_id': image_id, 'scale': scale, 'test': '', 'img_name': image_name, 'ann': ann}
        data_input.update({'meta': meta})
        return data_input

    def __len__(self):
        return len(self.anns)
