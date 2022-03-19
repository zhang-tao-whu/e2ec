from torch.utils.data.dataloader import default_collate
import torch


def collate_batch(batch):
    data_input = {}
    inp = {'inp': default_collate([b['inp'] for b in batch])}
    meta = default_collate([b['meta'] for b in batch])
    data_input.update(inp)
    data_input.update({'meta': meta})

    if 'test' in meta:
        return data_input

    #collate detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])
    max_len = torch.max(meta['ct_num'])
    batch_size = len(batch)
    wh = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, max_len], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
    ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1
        ct_img_idx[i, :meta['ct_num'][i]] = i

    if max_len != 0:
        wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
        # reg[ct_01] = torch.Tensor(sum([b['reg'] for b in batch], []))
        ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
        ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))
    detection = {'ct_hm': ct_hm, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01, 'ct_img_idx': ct_img_idx}
    data_input.update(detection)

    #collate sementation
    num_points_per_poly = 128
    img_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    can_gt_polys = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
    keyPointsMask = torch.zeros([batch_size, max_len, num_points_per_poly], dtype=torch.float)

    if max_len != 0:
        img_gt_polys[ct_01] = torch.Tensor(sum([b['img_gt_polys'] for b in batch], []))
        can_gt_polys[ct_01] = torch.Tensor(sum([b['can_gt_polys'] for b in batch], []))
        keyPointsMask[ct_01] = torch.Tensor(sum([b['keypoints_mask'] for b in batch], []))
    data_input.update({'img_gt_polys': img_gt_polys, 'can_gt_polys': can_gt_polys,
                       'keypoints_mask': keyPointsMask})

    return data_input
