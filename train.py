import cv2 as cv
import trimesh
import logging
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
from pyhocon import ConfigFactory, HOCONConverter
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import Dataset
from models.fields import NeRF, RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer

class Trainer:

    def __init__(self, conf, resume=False):
        self.conf = conf
        self.device = torch.device('cuda')
        self.base_exp_dir = conf.get_string('general.base_exp_dir')
        output = Path(self.base_exp_dir)
        if not resume and (output / 'checkpoints').exists():
            raise FileExistsError('Use --resume or select a new --output directory')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'effective.conf').write_text(HOCONConverter.to_hocon(conf))
        args = argparse.Namespace(scene_num_for_train=conf.get_int('dataset.block_index'), radar_guided_rate=conf.get_float('train.radar_guided_rate'))
        self.dataset = Dataset(conf['dataset'], args)
        self.iter_step = 0
        self.image_perm = None
        self.end_iter = conf.get_int('train.end_iter')
        self.batch_size = conf.get_int('train.batch_size')
        self.radar_sup_n = conf.get_int('train.radar_sup_n')
        self.learning_rate = conf.get_float('train.learning_rate')
        self.learning_rate_alpha = conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = conf.get_float('train.warm_up_end')
        self.anneal_end = conf.get_float('train.anneal_end')
        self.igr_weight = conf.get_float('train.igr_weight')
        self.mask_weight = conf.get_float('train.mask_weight')
        self.use_white_bkgd = conf.get_bool('train.use_white_bkgd')
        self.validate_resolution_level = conf.get_int('train.validate_resolution_level')
        self.nerf_outside = NeRF(**conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**conf['model.rendering_network']).to(self.device)
        parameters = []
        for module in [self.nerf_outside, self.sdf_network, self.deviation_network, self.color_network]:
            parameters += list(module.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.renderer = NeuSRenderer(self.nerf_outside, self.sdf_network, self.deviation_network, self.color_network, **conf['model.neus_renderer'])
        if resume:
            checkpoint = sorted((output / 'checkpoints').glob('ckpt_*.pth'))[-1]
            self.load_checkpoint(checkpoint)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            factor = self.iter_step / self.warm_up_end
        else:
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - self.learning_rate_alpha) + self.learning_rate_alpha
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate * factor

    def train(self, stop_iter=None):
        stop_iter = self.end_iter if stop_iter is None else stop_iter
        if not self.iter_step <= stop_iter <= self.end_iter:
            raise ValueError('stop_iter must lie between the current step and end_iter')
        self.update_learning_rate()
        if self.image_perm is None:
            self.image_perm = self.get_image_perm()
        for _ in tqdm(range(self.iter_step, stop_iter)):
            view = self.image_perm[self.iter_step % len(self.image_perm)]
            data = self.dataset.gen_rd_pcd_scene_mask_guided_perturb_rays_at_unified_large_small(view, self.batch_size, using_radar=True)
            (rays_o, rays_d) = (data[:, :3], data[:, 3:6])
            (true_rgb, mask) = (data[:, 6:9], data[:, 9:10])
            (near, far) = self.dataset.near_far_in_aerial_scene(rays_o, rays_d)
            background = torch.ones([1, 3]) if self.use_white_bkgd else None
            mask = (mask > 0.5).float() if self.mask_weight > 0.0 else torch.ones_like(mask)
            mask_sum = mask.sum() + 1e-05
            result = self.renderer.render(rays_o, rays_d, near, far, background_rgb=background, cos_anneal_ratio=self.get_cos_anneal_ratio())
            points = self.dataset.gen_pts_random(self.radar_sup_n)
            sdf = self.sdf_network.sdf(points)
            sdf_loss = F.l1_loss(sdf, torch.zeros_like(sdf), reduction='sum') / sdf.shape[0]
            color_loss = F.l1_loss((result['color_fine'] - true_rgb) * mask, torch.zeros_like(true_rgb), reduction='sum') / mask_sum
            mask_loss = F.binary_cross_entropy(result['weight_sum'].clip(0.001, 1 - 0.001), mask)
            loss = color_loss + self.igr_weight * result['gradient_error'] + self.mask_weight * mask_loss + sdf_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iter_step += 1
            if self.iter_step % self.conf.get_int('train.report_freq') == 0:
                row = dict(step=self.iter_step, loss=float(loss.detach()), color=float(color_loss.detach()), sdf=float(sdf_loss.detach()))
                with (Path(self.base_exp_dir) / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            if self.iter_step % self.conf.get_int('train.val_freq') == 0:
                self.validate_image()
            if self.iter_step % self.conf.get_int('train.val_mesh_freq') == 0:
                self.validate_block_mesh(world_space=True, resolution=512)
            self.update_learning_rate()
            if self.iter_step % len(self.image_perm) == 0:
                self.image_perm = self.get_image_perm()
            if self.iter_step % self.conf.get_int('train.save_freq') == 0 or self.iter_step == stop_iter:
                self.save_checkpoint()

    def save_checkpoint(self):
        state = dict(nerf=self.nerf_outside.state_dict(), sdf_network_fine=self.sdf_network.state_dict(), variance_network_fine=self.deviation_network.state_dict(), color_network_fine=self.color_network.state_dict(), optimizer=self.optimizer.state_dict(), iter_step=self.iter_step, image_perm=self.image_perm, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), conf=HOCONConverter.to_hocon(self.conf))
        folder = Path(self.base_exp_dir) / 'checkpoints'
        folder.mkdir(exist_ok=True)
        target = folder / f'ckpt_{self.iter_step:06d}.pth'
        temp = target.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(target)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        if state['conf'] != HOCONConverter.to_hocon(self.conf):
            raise ValueError('Resume with the same effective configuration')
        for (name, key) in [('nerf_outside', 'nerf'), ('sdf_network', 'sdf_network_fine'), ('deviation_network', 'variance_network_fine'), ('color_network', 'color_network_fine')]:
            getattr(self, name).load_state_dict(state[key])
        self.optimizer.load_state_dict(state['optimizer'])
        self.iter_step = state['iter_step']
        self.image_perm = state['image_perm'].cuda()
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        (rays_o, rays_d) = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        (H, W, _) = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        out_rgb_fine = []
        out_normal_fine = []
        for (rays_o_batch, rays_d_batch) in zip(rays_o, rays_d):
            (near, far) = self.dataset.near_far_in_aerial_scene(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far, cos_anneal_ratio=self.get_cos_anneal_ratio(), background_rgb=background_rgb)

            def feasible(key):
                return key in render_out and render_out[key] is not None
            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)), np.concatenate([img_fine[..., i], self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)), normal_img[..., i])

    def validate_block_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        (vertices, triangles) = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles)
        if world_space:
            mesh.apply_transform(self.dataset.scale_mats_np[0])
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        logging.info('End')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='configs/suzhou_s1.conf')
    parser.add_argument('--data-dir')
    parser.add_argument('--output')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--block', type=int)
    parser.add_argument('--stop-iter', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export', action='store_true')
    options = parser.parse_args()
    conf = ConfigFactory.parse_file(options.conf)
    if options.data_dir:
        conf.put('dataset.data_dir', options.data_dir)
    if options.output:
        conf.put('general.base_exp_dir', options.output)
    if options.block is not None:
        conf.put('dataset.block_index', options.block)
    torch.cuda.set_device(options.gpu)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    seed = conf.get_int('train.seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainer = Trainer(conf, resume=options.resume or options.export)
    if options.export:
        trainer.validate_block_mesh(world_space=True, resolution=512)
    else:
        trainer.train(options.stop_iter)
if __name__ == '__main__':
    main()
