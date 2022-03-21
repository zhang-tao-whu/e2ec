import torch.nn as nn
from .snake import Snake
from .utils import prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature
import torch


class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4.):
        super(Evolution, self).__init__()
        assert evole_ietr_num >= 1
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.iter = evole_ietr_num - 1
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False):
        if ignore:
            return i_it_poly * self.ro
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        c_it_poly = c_it_poly * self.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        offset = snake(init_input).permute(0, 2, 1)
        i_poly = i_it_poly * self.ro + offset * stride
        return i_poly

    def foward_train(self, output, batch, cnn_feature):
        ret = output
        init = self.prepare_training(output, batch)
        py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, init['img_init_polys'],
                                   init['can_init_polys'], init['py_ind'], stride=self.evolve_stride)
        py_preds = [py_pred]
        for i in range(self.iter):
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred,
                                       init['py_ind'], stride=self.evolve_stride)
            py_preds.append(py_pred)
        ret.update({'py_pred': py_preds, 'img_gt_polys': init['img_gt_polys'] * self.ro})
        return output

    def foward_test(self, output, cnn_feature, ignore):
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py = self.evolve_poly(self.evolve_gcn, cnn_feature, img_init_polys, init['can_init_polys'], init['py_ind'],
                                  ignore=ignore[0], stride=self.evolve_stride)
            pys = [py, ]
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['py_ind'],
                                      ignore=ignore[i + 1], stride=self.evolve_stride)
                pys.append(py)
            ret.update({'py': pys})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:
            self.foward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.foward_test(output, cnn_feature, ignore=ignore)
        return output

