from .base import commen, data, model, train, test
import numpy as np

data.scale = np.array([896, 384])
data.input_w, input_h = (896, 384)
data.scale_range = np.arange(0.4, 1.0, 0.1)

model.heads['ct_hm'] = 7

train.dataset = 'kitti_train'
train.optimizer['gamma'] = 0.25
train.batch_size = 64
train.num_workers = 64

test.test_rescale = 0.5
test.dataset = 'kitti_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
