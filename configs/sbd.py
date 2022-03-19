from .base import commen, data, model, train, test

data.scale = None
data.test_scale = (512, 512)

model.heads['ct_hm'] = 20

train.dataset = 'sbd_train'

test.dataset = 'sbd_val'
test.with_nms = False

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test