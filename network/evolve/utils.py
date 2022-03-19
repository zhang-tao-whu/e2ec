import torch

def collect_training(poly, ct_01):
    batch_size = ct_01.size(0)
    poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
    return poly

def prepare_training(ret, batch, ro):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'img_gt_polys': collect_training(batch['img_gt_polys'], ct_01)})
    init.update({'img_init_polys': ret['poly_coarse'].detach() / ro})
    can_init_polys = img_poly_to_can_poly(ret['poly_coarse'].detach() / ro)
    init.update({'can_init_polys': can_init_polys})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})
    init.update({'py_ind': init['py_ind']})
    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature

def prepare_testing_init(polys, ro):
    polys = polys / ro
    can_init_polys = img_poly_to_can_poly(polys)
    img_init_polys = polys
    ind = torch.zeros((img_init_polys.size(0), ), dtype=torch.int32, device=img_init_polys.device)
    init = {'img_init_polys': img_init_polys, 'can_init_polys': can_init_polys, 'py_ind': ind}
    return init
