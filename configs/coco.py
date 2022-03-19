from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 80

train.batch_size = 24
train.epoch = 140
train.dataset = 'coco_train'

test.dataset = 'coco_val'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test