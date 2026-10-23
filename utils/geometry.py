import numpy as np
import torch

def resolve_plane_equation(point, normal_vec):
    A = normal_vec[0]
    B = normal_vec[1]
    C = normal_vec[2]
    D = -1 * np.sum(normal_vec * point)
    return np.asarray([A, B, C, D])

def cal_point2plane_dis(point, plane):
    (A, B, C, D) = (plane[0], plane[1], plane[2], plane[3])
    d = torch.abs(A * point[:, 0] + B * point[:, 1] + C * point[:, 2] + D) / torch.sqrt(A ** 2 + B ** 2 + C ** 2)
    return d

def cal_vecs_angle(vec1, vec2):
    dot_product = torch.sum(torch.mul(vec1, torch.unsqueeze(vec2, 0)), dim=1)
    mag_1 = torch.linalg.norm(vec1, dim=1)
    mag_2 = torch.linalg.norm(vec2)
    cos_theta = dot_product / (mag_1 * mag_2)
    angle_rad = torch.acos(cos_theta)
    return torch.pi - angle_rad

def synthesize_mask_rd_randchoose_pix(mask, pix_coord_rd, pix_rand_choose, batch_size, ratio, is_perturb=False):
    device = pix_coord_rd.device
    (H, W) = mask.shape
    mask_flat = mask.to(pix_coord_rd.dtype)
    if is_perturb:
        dx = torch.randint(low=-7, high=8, size=pix_coord_rd[0, :].shape, device=pix_coord_rd.device, dtype=torch.int32)
        dy = torch.randint(low=-7, high=8, size=pix_coord_rd[1, :].shape, device=pix_coord_rd.device, dtype=torch.int32)
        pix_coord_rd[0, :] = pix_coord_rd[0, :] + dx
        pix_coord_rd[1, :] = pix_coord_rd[1, :] + dy
        pix_coord_rd[0, :] = torch.clamp(pix_coord_rd[0, :], 0, W - 1)
        pix_coord_rd[1, :] = torch.clamp(pix_coord_rd[1, :], 0, H - 1)
    rd_x = pix_coord_rd[0, :]
    rd_y = pix_coord_rd[1, :]
    valid_ind = (rd_x >= 0) & (rd_x < W) & (rd_y >= 0) & (rd_y < H)
    x = rd_x[valid_ind]
    y = rd_y[valid_ind]
    valid_pix_coord_rd = pix_coord_rd[:, valid_ind]
    is_inner_mask = mask[x.cpu(), y.cpu()]
    inner_pix_coord_rd = valid_pix_coord_rd[:, is_inner_mask]
    num_inner = inner_pix_coord_rd.shape[1]
    num_to_replace = int(batch_size * ratio)
    if num_inner >= num_to_replace and num_to_replace > 0:
        indices = torch.randperm(num_inner, device=device)[:num_to_replace]
        sel_inner_pix_coord_rd = inner_pix_coord_rd[:, indices]
        replace_indices = torch.randperm(batch_size, device=device)[:num_to_replace]
        pix_rand_choose[:, replace_indices] = sel_inner_pix_coord_rd
    elif 0 < num_inner < num_to_replace:
        indices = torch.randperm(num_inner, device=device)
        replace_indices = torch.randperm(batch_size, device=device)[:num_inner]
        pix_rand_choose[:, replace_indices] = inner_pix_coord_rd[:, indices]
    else:
        pass
    return (pix_rand_choose, num_inner)
