from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 80

train.optimizer = {'name': 'sgd', 'lr': 1e-4, 'weight_decay': 1e-4,
                   'milestones': [150, ], 'gamma': 0.1}
train.batch_size = 24
train.epoch = 160
train.dataset = 'coco_train'

test.dataset = 'coco_val'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test