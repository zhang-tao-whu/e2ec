from .base import Dataset
import os
import cv2
import numpy as np

class CityscapesCocoDataset(Dataset):

    def read_original_data(self, anno, image_path):
        img = cv2.imread(image_path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in instance['segmentation']] for instance in anno
                          if not isinstance(instance['segmentation'], dict)]
        cls_ids = [self.json_category_id_to_continuous_id[instance['category_id']] for instance in anno]
        return img, instance_polys, cls_ids

    def process_info(self, ann):
        image_id = ann
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        file_name = self.coco.loadImgs(int(image_id))[0]['file_name']
        city = file_name.split('_')[0]
        image_path = os.path.join(self.data_root, city, file_name)
        return ann, image_path, image_id