import torchvision
import os
import cv2
from ..train.utils import augment

class Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, data_root, split, cfg):
        super(Dataset, self).__init__(data_root, ann_file)
        self.ids = sorted(self.ids)
        self.data_root = data_root
        self.split = split
        self.cfg = cfg

    def process_info(self, img_id):
        image_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(self.data_root, image_name)
        return path, image_name

    def read_original_data(self, path):
        img = cv2.imread(path)
        return img

    def __getitem__(self, index):
        img_id = self.ids[index]
        path, image_name = self.process_info(img_id)
        img = self.read_original_data(path)

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            augment(
                img, self.split,
                self.cfg.data.data_rng, self.cfg.data.eig_val, self.cfg.data.eig_vec,
                self.cfg.data.mean, self.cfg.data.std, self.cfg.commen.down_ratio,
                self.cfg.data.input_h, self.cfg.data.input_w, self.cfg.data.scale_range,
                self.cfg.data.scale, self.cfg.test.test_rescale, self.cfg.data.test_scale
            )

        ret = {'inp': inp}

        meta = {'center': center, 'img_id': img_id, 'scale': scale, 'test': '', 'img_name': image_name}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.ids)