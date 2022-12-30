from .base import commen, data, model, train, test
import numpy as np

data.scale = np.array([800, 800])
data.input_w, data.input_h = (800, 800)

model.heads['ct_hm'] = 8

train.dataset = 'cityscapes_train'
train.optimizer['milestones'] = [80, 120, 150, ]
train.batch_size = 32
train.num_workers = 32
train.epoch = 200

test.test_rescale = 0.85
test.dataset = 'cityscapes_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test
