import os
from glob import glob
import cv2 as cv
import numpy as np
import torch
from plyfile import PlyData
from utils.geometry import resolve_plane_equation, cal_point2plane_dis, cal_vecs_angle, synthesize_mask_rd_randchoose_pix

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(' ') for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return (intrinsics, pose)

class Dataset:

    def __init__(self, conf, args):
        super(Dataset, self).__init__()
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.load_type = 'ram'
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.partition_masks_folder = conf.get_string('partition_folder')
        self.scene_num_for_train = args.scene_num_for_train
        self.radar_guided_rate = args.radar_guided_rate
        self.scene_size = conf.get_float('scene_size')
        self.world_height_max = conf.get_float('world_height_max')
        self.world_ground = conf.get_float('world_ground')
        self.aerial_extand = conf.get_bool('aerial_extand')
        self.aerial_extand_sz = torch.tensor(conf.get_float('aerial_extand_sz')).type(torch.float32).cuda()
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.jpg')))
        self.choose_idx = np.asarray(conf.get_list('view_indices'), dtype=np.int64)
        if len(set(self.choose_idx)) != len(self.choose_idx) or np.any(self.choose_idx < 0) or np.any(self.choose_idx >= len(self.images_lis)):
            raise ValueError('view_indices must be unique zero-based image/camera indices')
        self.images_lis = np.array(self.images_lis)[self.choose_idx]
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        self.masks_lis = np.array(self.masks_lis)[self.choose_idx]
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.stack([cv.imread(im_name, 0) for im_name in self.masks_lis]) / 256.0
        self.world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.choose_idx]
        self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.choose_idx]
        scale = self.scale_mats_np[0]
        expected_scale = np.diag([scale[0, 0], scale[0, 0], scale[0, 0], 1.0])
        if scale[0, 0] <= 0 or not np.allclose(scale, expected_scale) or (not all((np.allclose(s, scale) for s in self.scale_mats_np))):
            raise ValueError('This release requires the original shared, positive uniform scale_mat without rotation or translation')
        self.intrinsics_all = []
        self.pose_all = []
        for (scale_mat, world_mat) in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            (intrinsics, pose) = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks = torch.from_numpy(self.masks_np.astype(bool)).cpu()
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        self.world_mats = torch.Tensor(self.world_mats_np).to(self.device)
        self.scale_mats = torch.Tensor(self.scale_mats_np).to(self.device)
        (self.H, self.W) = (self.images_np.shape[1], self.images_np.shape[2])
        if self.H != self.W:
            raise ValueError('The original guided sampler requires square input images')
        self.image_pixels = self.H * self.W
        self.is_mega = conf.get_bool('partitioned', default=False)
        if self.is_mega:
            self.scene_paras = torch.load(os.path.join(self.data_dir, self.partition_masks_folder, 'params.pt'))
            self.scene_block_num = self.scene_paras['centroids'].shape[0]
            self.partition_masks_lis = np.array(sorted(glob(os.path.join(self.data_dir, self.partition_masks_folder, str(self.scene_num_for_train), '*.pt'))))[self.choose_idx]
            self.partition_masks = [torch.load(f, map_location='cpu') for f in self.partition_masks_lis]
        else:
            self.scene_paras = torch.load(os.path.join(self.data_dir, 'scene_cfg.pt'))
        world_ground_center = np.asarray([[0, 0, self.world_ground]])
        world_height = np.asarray([[0, 0, self.world_height_max]])
        world_ground_center_homo = np.hstack((world_ground_center, np.ones((world_ground_center.shape[0], 1))))
        world_height_homo = np.hstack((world_height, np.ones((world_height.shape[0], 1))))
        self.implicit_world_ground_center = np.squeeze((np.linalg.inv(self.scale_mats_np[0]) @ world_ground_center_homo.T)[:-1])
        self.implicit_world_height = np.squeeze((np.linalg.inv(self.scale_mats_np[0]) @ world_height_homo.T)[:-1])
        self.implicit_world_ground_normal_vec = np.squeeze(self.implicit_world_height - self.implicit_world_ground_center)
        self.implicit_ground_plane = torch.from_numpy(resolve_plane_equation(point=self.implicit_world_ground_center, normal_vec=self.implicit_world_ground_normal_vec)).type(torch.float32).cuda()
        self.implicit_height_plane = torch.from_numpy(resolve_plane_equation(point=self.implicit_world_height, normal_vec=self.implicit_world_ground_normal_vec)).type(torch.float32).cuda()
        self.implicit_world_ground_normal_vec = torch.from_numpy(self.implicit_world_ground_normal_vec).type(torch.float32).cuda()
        rd_pcd_reg_dir = os.path.join(self.data_dir, conf.get_string('radar_file'))
        ply_data = PlyData.read(rd_pcd_reg_dir)
        vertices = ply_data['vertex']
        pcd_reg = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T
        del ply_data, vertices
        if self.is_mega:
            scene_block_center = self.scene_paras['centroids'][self.scene_num_for_train]
            x_bound = torch.asarray([self.scene_paras['pose_scale_factor'] * scene_block_center[0] - self.scene_size / 2, self.scene_paras['pose_scale_factor'] * scene_block_center[0] + self.scene_size / 2]).numpy()
            y_bound = torch.asarray([self.scene_paras['pose_scale_factor'] * scene_block_center[1] - self.scene_size / 2, self.scene_paras['pose_scale_factor'] * scene_block_center[1] + self.scene_size / 2]).numpy()
            z_bound = [-5, self.world_height_max]
        else:
            scene_center = self.scene_paras['origin']
            x_bound = torch.asarray([self.scene_paras['pose_scale_factor'] * scene_center[0] - self.scene_size / 2, self.scene_paras['pose_scale_factor'] * scene_center[0] + self.scene_size / 2]).numpy()
            y_bound = torch.asarray([self.scene_paras['pose_scale_factor'] * scene_center[1] - self.scene_size / 2, self.scene_paras['pose_scale_factor'] * scene_center[1] + self.scene_size / 2]).numpy()
            z_bound = [-5, self.world_height_max]
        pcd_reg = pcd_reg[(pcd_reg[:, 0] >= x_bound[0]) & (pcd_reg[:, 0] <= x_bound[1]) & (pcd_reg[:, 1] >= y_bound[0]) & (pcd_reg[:, 1] <= y_bound[1]) & (pcd_reg[:, 2] >= z_bound[0]) & (pcd_reg[:, 2] <= z_bound[1])]
        pcd_reg_homo = np.hstack((pcd_reg, np.ones((pcd_reg.shape[0], 1))))
        self.pts = torch.from_numpy((np.linalg.inv(self.scale_mats_np[0]) @ pcd_reg_homo.T).T[:, :-1]).type(torch.float32).cuda()
        self.pts_n = self.pts.shape[0]
        if self.pts_n == 0:
            raise ValueError('No SAR points remain in the training region')
        self.object_bbox_min = np.array([x_bound[0], y_bound[0], z_bound[0]]) / self.scene_paras['pose_scale_factor']
        self.object_bbox_max = np.array([x_bound[1], y_bound[1], z_bound[1]]) / self.scene_paras['pose_scale_factor']

    def gen_rays_at(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        (pixels_x, pixels_y) = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)
        return (rays_o.transpose(0, 1), rays_v.transpose(0, 1))

    def gen_rd_pcd_scene_mask_guided_perturb_rays_at_unified_large_small(self, img_idx, batch_size, using_radar=False):
        img_pix_ratio = self.radar_guided_rate
        if self.is_mega:
            mask = self.masks[img_idx, ...] & self.partition_masks[img_idx]
        else:
            mask = self.masks[img_idx, ...]
        coords = torch.nonzero(mask)
        if len(coords) < batch_size:
            raise ValueError(f'Mask only contains {len(coords)} True elements, but required batch_size is {batch_size}')
        sel_indices = torch.randperm(coords.size(0), device=coords.device)[0:batch_size]
        pixels_x_random = coords[sel_indices, 0].to(self.device)
        pixels_y_random = coords[sel_indices, 1].to(self.device)
        pixels_xy_random = torch.stack([pixels_x_random, pixels_y_random])
        if using_radar:
            radar_pts_homo = torch.hstack([self.pts, torch.ones((self.pts.shape[0], 1))])
            pix_coord = self.world_mats[img_idx] @ self.scale_mats[img_idx] @ radar_pts_homo.T
            pix_coord[0:3, :] = pix_coord[0:3, :] / pix_coord[2, :]
            pix_coord_ceil = torch.ceil(pix_coord[[1, 0], :]).type(torch.int64)
            (pixels_xy, _) = synthesize_mask_rd_randchoose_pix(mask, pix_coord_ceil, pixels_xy_random, batch_size, img_pix_ratio, is_perturb=True)
        else:
            pixels_xy = pixels_xy_random
        pixels_x = pixels_xy[0, :].to(self.device)
        pixels_y = pixels_xy[1, :].to(self.device)
        color = self.images[img_idx.cpu()][pixels_x.cpu(), pixels_y.cpu()]
        mask = self.masks[img_idx.cpu()][pixels_x.cpu(), pixels_y.cpu()].unsqueeze(1)
        p = torch.stack([pixels_y, pixels_x, torch.ones_like(pixels_y)], dim=-1).float()
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask], dim=-1).cuda()

    def near_far_in_aerial_scene(self, rays_o, rays_d):
        o2height_plane = cal_point2plane_dis(point=rays_o.clone().detach(), plane=self.implicit_height_plane)
        o2ground_plane = cal_point2plane_dis(point=rays_o.clone().detach(), plane=self.implicit_ground_plane)
        angle_between_d_normal = cal_vecs_angle(vec1=rays_d.clone().detach(), vec2=self.implicit_world_ground_normal_vec)
        near = torch.unsqueeze(o2height_plane / torch.cos(angle_between_d_normal), dim=1)
        far = torch.unsqueeze(o2ground_plane / torch.cos(angle_between_d_normal), dim=1)
        if self.aerial_extand:
            near = near - self.aerial_extand_sz
            far = far + self.aerial_extand_sz
        return (near, far)

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return cv.resize(img, (self.W // resolution_level, self.H // resolution_level)).clip(0, 255)

    def gen_pts_random(self, sup_n=512):
        pts_choose_idx = np.random.randint(0, self.pts_n, size=sup_n)
        pts_choose = self.pts[pts_choose_idx]
        return pts_choose
