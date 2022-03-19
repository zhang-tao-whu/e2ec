import torch
from .utils import decode_ct_hm, clip_to_image, get_gcn_feature

class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)
        poly_num = ct_polys.size(0)

        feature = self.trans_feature(feature)

        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys

class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.refine = Refine(c_in=c_in, num_point=num_point, stride=coarse_stride)

    def train_decode(self, data_input, output, cnn_feature):
        wh_pred = output['wh']
        ct_01 = data_input['ct_01'].bool()
        ct_ind = data_input['ct_ind'][ct_01]
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        _, _, height, width = data_input['ct_hm'].size()
        ct_x, ct_y = ct_ind % width, ct_ind // width

        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)

        ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
        ct = torch.cat([ct_x, ct_y], dim=1)

        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        coarse_polys = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone())

        init_polys = clip_to_image(init_polys, cnn_feature.size(2), cnn_feature.size(3))
        coarse_polys = clip_to_image(coarse_polys, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        return

    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride)
        valid = detection[0, :, 2] >= min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})

        img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)
        poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
        coarse_polys = clip_to_image(poly_coarse, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * self.down_sample})
        output.update({'detection': detection})
        return

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            self.train_decode(data_input, output, cnn_feature)
        else:
            self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform)

