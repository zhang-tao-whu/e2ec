import torch

def box_iou(boxes1, boxes2):
    n = boxes1.size(0)
    boxes1 = boxes1.unsqueeze(0).expand(n, n, 4)
    boxes2 = boxes2.unsqueeze(1).expand(n, n, 4)
    left = torch.cat([boxes1[:, :, :2].unsqueeze(3), boxes2[:, :, :2].unsqueeze(3)], dim=3)
    left = torch.max(left, dim=3)[0]
    right = torch.cat([boxes1[:, :, 2:].unsqueeze(3), boxes2[:, :, 2:].unsqueeze(3)], dim=3)
    right = torch.min(right, dim=3)[0]
    wh = right - left
    wh = torch.clamp(wh, 0, 64000000)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    boxes1_wh = boxes1[:, :, 2:] - boxes1[:, :, :2]
    boxes1_area = boxes1_wh[:, :, 0] * boxes1_wh[:, :, 1]
    boxes2_wh = boxes2[:, :, 2:] - boxes2[:, :, :2]
    boxes2_area = boxes2_wh[:, :, 0] * boxes2_wh[:, :, 1]
    ious = inter_area / (boxes1_area + boxes2_area - inter_area)

    return torch.clamp(ious, 0, 1)

def fast_nms(boxes, scores, NMS_threshold=0.7):
    '''
    Arguments:
        boxes (Tensor[N, 4])
        scores (Tensor[N, 1])
    Returns:
        Fast NMS results
    '''
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx]
    iou = box_iou(boxes, boxes)
    iou = iou.triu_(diagonal=1)
    keep = iou.max(dim=0)[0] < NMS_threshold
    return keep, idx

def post_process(output):
    detection = output['detection']
    score = detection[:, 2].detach()
    label = detection[:, 3].detach()
    last_py = output['py'][-1].detach()
    if len(last_py) == 0:
        return 0
    off_max = torch.max(last_py)
    boxes = torch.cat([torch.min(last_py, dim=1)[0], torch.max(last_py, dim=1)[0]], dim=1)
    boxes = (boxes.permute(1, 0) + label * off_max).permute(1, 0)
    keep, idx = fast_nms(boxes, score)
    detection = detection[idx][keep]
    ret_p = last_py[idx][keep]
    output.update({"detection": detection})
    output['py'].append(ret_p)
    return 0
