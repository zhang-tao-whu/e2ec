import torch

def compute_num(offset_0, offset_1, thre=3):
    offset_0_front = torch.roll(offset_0, shifts=1, dims=1)
    offset_1_front = torch.roll(offset_1, shifts=1, dims=1)
    offset_0_front_2 = torch.roll(offset_0, shifts=2, dims=1)
    offset_1_front_2 = torch.roll(offset_1, shifts=2, dims=1)
    cos_0 = torch.sum(offset_0 * offset_0_front, dim=2)
    cos_1 = torch.sum(offset_1 * offset_1_front, dim=2)
    cos_0_2 = torch.sum(offset_0 * offset_0_front_2, dim=2)
    cos_1_2 = torch.sum(offset_1 * offset_1_front_2, dim=2)
    cos_0 = ((cos_0 < -0.1) & (cos_0_2 > 0.1)).to(torch.int)
    cos_1 = ((cos_1 < -0.1) & (cos_1_2 > 0.1)).to(torch.int)
    nums = (torch.sum(cos_1, dim=1) - torch.sum(cos_0, dim=1) >= thre).to(torch.int)
    nums = nums.unsqueeze(1).unsqueeze(2).expand(offset_0.size(0), offset_0.size(1), offset_0.size(2))
    return nums

def post_process(output):
    end_py = output['py'][-1].detach()
    gcn_py = output['py'][-2].detach()
    
    if len(end_py) == 0:
        return 0
    
    offset_1 = end_py - torch.roll(end_py, shifts=1, dims=1)
    offset_0 = gcn_py - torch.roll(gcn_py, shifts=1, dims=1)
    nokeep = compute_num(offset_0, offset_1)
    end_poly = end_py * (1 - nokeep) + gcn_py * nokeep
    output['py'].append(end_poly)
    return 0

