from .base import Dataset
import os

class CityscapesCocoTestDataset(Dataset):

    def process_info(self, img_id):
        image_name = self.coco.loadImgs(img_id)[0]['file_name']
        if '_' in image_name:
            city = image_name.split('_')[0]
            path = os.path.join(self.data_root, city, image_name)
        else:
            path = os.path.join(self.data_root, image_name)
        return path, image_name
