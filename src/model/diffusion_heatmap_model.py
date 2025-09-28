# Copyright (c) Phigent Robotics. All rights reserved.

"""
# from model.builder import DETECTORS
# from model.builder import HEADS, build_loss
from model.backbone import get as get_backbone
from model.head import get as get_head
"""
import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np
import json
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.core import bbox3d2result
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.models import BACKBONES
from mmdet3d.models.builder import HEADS, build_loss
import mmcv
from collections import OrderedDict
import model.ops.ip_basic as ip_basic
from model.backbone import get as get_backbone
# It is needed for head registration
from model.head import get as get_head
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF


@DETECTORS.register_module()
class Diffusion_Heatmap_Model(nn.Module):
    def __init__(
            self,
            args,
            depth_backbone_cfg=dict(type='mmbev_res50'),
            depth_head_cfg=None,
            norm_cfg=dict(type='BN'),
            depth_backbone=None,
            depth_head=None,
            center_depth_head=None,
            dense_pts_voxel_encoder=None,
            bev_fusion=None,
            pts_seg_head=None,
            lidar_stream_pretrain=None,
            cam_stream_pretrain=None,
            depth_stream_pretrain=None,
            freeze_camera_branch=False,
            freeze_lidar_branch=False,
            not_freeze_pts_bbox_head=False,
            freeze_depth_branch=False,
            with_instance_mask=False,
            ip_basic=False,
            depth_keys='all',
            **kwargs
    ):

        super(Diffusion_Heatmap_Model, self).__init__()
        if depth_head_cfg is None:
            depth_head_cfg = dict(type='DDIMDepthRefine2',
                                  in_channels=[64, 128, 256, 512],  # resnet18
                                  depth_feature_dim=16,  # deep based 16 otherwise 2 for traditional
                                  loss_cfgs=[
                                      dict(loss_func='l1_depth_loss', name='depth_loss', weight=0.2, pred_indices=0,
                                           gt_indices=0),
                                      dict(loss_func='l1_depth_loss', name='blur_depth_loss', weight=0.1,
                                           pred_indices=1, gt_indices=0),
                                  ],
                                  diff_type=args.diffusion_type,
                                  )
        """
        self.depth_backbone = mmbev_resnet.ResNetForMMBEV(3, num_layer=[2, 2, 2, 2], num_channels=[64, 128, 256, 512],
                                                          stride=[2, 2, 2, 2],
                                                          backbone_output_ids=None, norm_cfg=dict(type='BN'),
                                                          with_cp=False, block_type='Basic', )
        """
        self.args = args

        if depth_backbone is None:
            # self.depth_backbone = BACKBONES.build(depth_backbone_cfg)
            self.depth_backbone = get_backbone(args)
            self.depth_backbone = self.depth_backbone()

        if depth_head is None:
            try:
                inference_steps = self.args.inference_steps
                num_train_timesteps = self.args.num_train_timesteps
            except:
                print("there is no prefix inference steps loaded, make it to defualt 20/1000")
                exit(1)
                inference_steps = 20
                num_train_timesteps = 1000
            if args.head_specify is not None:
                depth_head_cfg = dict(type=args.head_specify,
                                      in_channels=[64, 128, 256, 512],  # resnet18
                                      inference_steps=inference_steps,
                                      num_train_timesteps=num_train_timesteps,
                                      depth_feature_dim=16,  # deep based 16 otherwise 2 for traditional
                                      loss_cfgs=[
                                          dict(loss_func='l1_depth_loss', name='depth_loss', weight=0.2, pred_indices=0,
                                               gt_indices=0),
                                          dict(loss_func='l1_depth_loss', name='blur_depth_loss', weight=0.1,
                                               pred_indices=1, gt_indices=0),
                                      ],
                                      diff_type=args.diffusion_type,
                                      image_width=args.patch_width,
                                      image_height=args.patch_height,
                                      init_cfg=args,
                                      )
                self.head_specify = args.head_specify
                self.real_name = args.real_name
                self.dir_real = args.dir_real

                self.premeasured_num = args.premeasured_num

            else:
                self.head_specify = None

            if args.mink_pretrained_path:
                depth_head_cfg["mink_pretrained_path"] = args.mink_pretrained_path
            else:
                print("########No pretrained models########")
                # raise Exception
            if "Video" in self.head_specify:
                depth_head_cfg["video_fps"] = args.video_fps

            if args.image_model_path:
                depth_head_cfg["image_model_path"] = args.image_model_path

            self.depth_head = HEADS.build(depth_head_cfg)
        self.with_instance_mask = with_instance_mask
        self.ip_basic = ip_basic
        self.depth_keys = depth_keys

    def _extract_depth_ipbasic(self, img, depth_map, img_metas):
        B, C, imH, imW = img.shape
        depth_map = depth_map
        depth_map.clamp_(0, 100)
        depth_map = depth_map.view(B, *depth_map.shape[-2:]).cpu()
        ret = []
        for i in range(B):
            depth_map_i = depth_map[i].numpy()
            dense_depth_map_i = ip_basic.fill_in_multiscale(depth_map_i)
            # dense_depth_map_i = ip_basic.fill_in_fast(depth_map_i, blur_kernel_size=3)
            ret.append(dense_depth_map_i)
        dense_depth = [torch.tensor(it, device=img.device) for it in ret]
        dense_depth = torch.stack(dense_depth)
        dense_depth = dense_depth.view(B, N, -1, *depth_map.shape[-2:])
        return dense_depth

    def extract_depth(self, overshot, heatmap, data_3d, freq_level, ap_coord, return_loss, img_metas, weight_map=None,
                      instance_masks=None, **kwargs):
        if self.ip_basic:
            return self._extract_depth_ipbasic(overshot, heatmap, img_metas)
        B, C, imH, imW = overshot.shape
        # print("here1: ", overshot.shape)
        # here1:  torch.Size([20, 3, 16, 16])
        kwargs['freq_level'] = freq_level
        kwargs['imH'] = imH
        kwargs['imW'] = imW

        overshot = overshot.view(B, C, imH, imW)
        fp = self.depth_backbone(overshot)
        # print("here3: ", len(fp), fp[0].size())
        # image: 4 torch.Size([2, 192, 80, 160])
        # video: 4 torch.Size([20, 192, 4, 4])
        # RGB vs. Float
        # overshot = overshot.view(B, 1, *overshot.shape[-2:])
        # heatmap = heatmap.view(B, 1, *heatmap.shape[-2:])
        if "A_P_Joint" in self.head_specify:
            num_output_type = 2  # Amp, Phase
            heatmap = heatmap.view(B * num_output_type, *heatmap.shape[-3:])
        else:
            heatmap = heatmap.view(B, *heatmap.shape[-3:])
        # gt_depth_map = gt_depth_map.view(B, 1, *heatmap.shape[-2:]) if gt_depth_map is not None else None
        weight_map = weight_map.view(B, *heatmap.shape[-2:]) if weight_map is not None else None
        instance_masks = instance_masks.view(B, 1,
                                             *instance_masks.shape[-2:]) if instance_masks is not None else None

        assert instance_masks is None
        assert weight_map is None
        # depth_mask = depth_mask.view(*heatmap.shape)
        # print(overshot.shape)

        # Premeasured
        # print(premeasured_image.shape)
        # print(overshot)
        # print(premeasured_image)
        # print("premeasured_images not zero", premeasured_images[premeasured_images != 0])
        # print("premeasured_image0", premeasured_images.shape)

        # import matplotlib.pyplot as plt
        # from PIL import Image
        # cm = plt.get_cmap('plasma')
        # premeasured_images = premeasured_images[0, :, :, :].data.cpu().numpy()
        # print("overshot not zero", overshot[overshot != 0])
        # print("premeasured_images not zero", premeasured_images[premeasured_images != 0])
        # print("heatmap not zero", heatmap[heatmap != 0])
        # premeasured_images not zero [0.24705882 0.24705882 0.1882353  0.54901963 0.24705882 0.40392157
        #  0.02745098 0.24313726 0.2901961  0.24705882]
        # assert not np.all(premeasured_images == 0), "All elements are zero"

        # premeasured_images = 255.0 * np.transpose(premeasured_images, (1, 2, 0))

        # print("premeasured_image", premeasured_images.shape)
        # print("overshot", overshot.size())
        # print("heatmap", heatmap.size())

        # premeasured_images = Image.fromarray(premeasured_images, 'RGB')

        if "A_P_Joint" in self.head_specify:
            assert ap_coord != []

            premeasured_images, premeasured_coords = self.get_premeasured_joint_image(heatmap, heatmap.size(2),
                                                                                      heatmap.size(3), ap_coord.cpu())

            amp_premeasured_map = premeasured_images[::2]
            phase_premeasured_map = premeasured_images[1::2]
            amp_fp2 = self.depth_backbone(amp_premeasured_map)
            phase_fp2 = self.depth_backbone(phase_premeasured_map)
            fp2 = (amp_fp2, phase_fp2)
        elif "Video" in self.head_specify:
            premeasured_images, premeasured_coords = self.get_premeasured_image(heatmap, heatmap.size(2),
                                                                                heatmap.size(3), fixed_coord=True)
            fp2 = self.depth_backbone(premeasured_images)

        # elif self.real_name:
        #     premeasured_images, premeasured_coords = self.get_premeasured_image_real(heatmap, heatmap.size(2),
        #                                                                         heatmap.size(3), fixed_coord=True)
        #     fp2 = self.depth_backbone(premeasured_images)

        else:
            premeasured_images, premeasured_coords = self.get_premeasured_image(heatmap, heatmap.size(2),
                                                                                heatmap.size(3), fixed_coord=True)
            fp2 = self.depth_backbone(premeasured_images)

        kwargs['premeasured_coords'] = premeasured_coords

        ret = self.depth_head(fp, fp2, heatmap, data_3d, return_loss=return_loss,
                              weight_map=weight_map, instance_masks=instance_masks,
                              image=overshot, **kwargs)

        if "A_P_Joint" in self.head_specify:
            ret['premeasured_map'] = (amp_premeasured_map, phase_premeasured_map)
        elif "Refractive" in self.head_specify:
            ret['premeasured_map'] = premeasured_images
        else:
            ret['premeasured_map'] = premeasured_images

        # if return_loss:
        # return ret
        # depth = ret.view(B, -1, *ret.shape[-2:])
        # import pdb; pdb.set_trace();
        return ret

    def forward(self, sample):
        """Forward training function.
        Args:
            sample containing four keys:

            for key in sample.keys():
                print('key {}'.format(key))
                print(sample[key].shape)
            key rgb
            torch.Size([3, 3, 228, 304])
            key dep
            torch.Size([3, 1, 228, 304])
            key gt
            torch.Size([3, 1, 228, 304])
            key K
            torch.Size([3, 4])
            ==
            key overshot
            torch.Size([4, 3, 352, 1216])
            key heatmap
            torch.Size([4, 1, 352, 1216])

            depth_maps = []
            for sparse_map in sparse_depth:
                depth_map = np.asarray(sparse_map, dtype=np.float32)
                depth_map, _ = simple_depth_completion(depth_map)
                depth_maps.append(depth_map)
            depth_maps = np.stack(depth_maps)  # bs, h, w

        Returns:
            dict: Losses of different branches.
        """
        """
        for key in sample.keys():
            print('key {}'.format(key))
            print(sample[key].shape)
        """
        overshot_input = sample['overshot']
        heatmap_input = sample['heatmap']
        coords = sample['coords']
        feats = sample['feats']
        # labels = sample['labels']

        freq_level = sample['freq_level'] if 'freq_level' in sample.keys() else None

        ap_coord = sample['ap_coord'] if 'ap_coord' in sample.keys() else None

        data_3d = (coords, feats, None)

        # inds_recons = sample['inds_recons']
        # print("here2: ", len(overshot_input), overshot_input[0].shape)
        # here2:  20 torch.Size([3, 16, 16])
        output_dict = self.extract_depth(overshot_input, heatmap_input, data_3d, freq_level, ap_coord, img_metas=None,
                                         return_loss=True, weight_map=None, instance_masks=None)
        return output_dict

    def get_premeasured_image(self, heatmap, x_size, y_size, fixed_coord=False):
        total_indices = x_size * y_size
        premeasured_heatmap = []
        premeasured_coords = []
        fixed_coords = None

        patch_span = x_size // 20
        # patch_span = 10
        if patch_span == 0:
            patch_span = 2
        for batch_idx in range(heatmap.size(0)):
            tmp_heatmap = np.zeros((x_size, y_size))
            # tmp_heatmap = np.full((x_size, y_size), fill_value=-100)

            sparse_indices = np.random.randint(0, total_indices, self.premeasured_num)
            """
            interval = total_indices // self.pre_measured_num  # Total number of indices divided by the number of sparse indices
            sparse_indices = [first_index + i * interval for i in range(self.pre_measured_num)]
            """
            if fixed_coord:
                if batch_idx == 0:
                    coordinates = []
                    for index in sparse_indices:
                        for dia_idx_x in range(patch_span):
                            for dia_idx_y in range(patch_span):
                                if x_size > (index // y_size) % x_size + dia_idx_x and y_size > (
                                        index % y_size) % y_size + dia_idx_y:
                                    coordinates.append(
                                        [(index // y_size) % x_size + dia_idx_x, (index % y_size) % y_size + dia_idx_y])

                    fixed_coords = coordinates
                else:
                    coordinates = fixed_coords
            else:
                coordinates = []
                for index in sparse_indices:
                    for dia_idx_x in range(patch_span):
                        for dia_idx_y in range(patch_span):
                            if x_size > (index // y_size) % x_size + dia_idx_x and y_size > (
                                    index % y_size) % y_size + dia_idx_y:
                                coordinates.append(
                                    [(index // y_size) % x_size + dia_idx_x, (index % y_size) % y_size + dia_idx_y])

                coordinates = np.array(coordinates, dtype=np.int64)

            for i, j in coordinates:
                if i < x_size and j < y_size:
                    tmp_heatmap[i][j] = heatmap[batch_idx, 0, i, j]

            # tmp_heatmap = torch.from_numpy(tmp_heatmap)
            # tmp_heatmap = Image.fromarray(tmp_heatmap.astype('float32'), mode='RGB')
            import matplotlib.pyplot as plt
            cm = plt.get_cmap('plasma')
            tmp_heatmap = (255.0 * cm(tmp_heatmap)).astype('uint8')
            tmp_heatmap = Image.fromarray(tmp_heatmap[:, :, :3], 'RGB')
            # tmp_heatmap = TF.to_tensor(np.array(tmp_heatmap))
            # tmp_heatmap = torch.from_numpy(tmp_heatmap)

            # tmp_heatmap = Image.fromarray(tmp_heatmap[:, :, :3], 'RGB')
            # path_save_premeasured = '00_premeasured.png'
            # tmp_heatmap.save(path_save_premeasured)

            tmp_heatmap = TF.to_tensor(np.array(tmp_heatmap))
            tmp_heatmap = TF.normalize(tmp_heatmap, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            coordinates = torch.from_numpy(np.array(coordinates))

            premeasured_heatmap.append(tmp_heatmap)
            premeasured_coords.append(coordinates)

        premeasured_heatmap = torch.stack(premeasured_heatmap, dim=0).cuda()
        premeasured_coords = torch.stack(premeasured_coords, dim=0).cuda()

        return premeasured_heatmap, premeasured_coords

    def get_premeasured_image_real(self, heatmap, x_size, y_size, fixed_coord=False):
        if self.real_name == 'apartment':
            json_name = 'exp_apartment-rssi-0817-codebook.json'
        elif self.real_name == '4321':
            json_name = 'exp_rssi-4321-retest-0824.json'
        else:
            raise Exception

        with open(self.dir_real + "data/" + json_name, 'r') as file:
            # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
            data = json.load(file)

        x_real = [entry["x"] for entry in data]
        y_real = [entry["y"] for entry in data]
        med_rssi_real = [entry["med_rssi"] for entry in data]
        med_rssi_real = np.clip(med_rssi_real, a_min=-57, a_max=-30)
        med_rssi_real = self.normalize_data(med_rssi_real, target_range=(0, 1))

        x_ratio = 4.04
        y_ratio = 2.47

        total_indices = x_size * y_size
        premeasured_heatmap = []
        premeasured_coords = []
        fixed_coords = None

        # patch_span = x_size // 20
        patch_span = 6
        if patch_span == 0:
            patch_span = 1
        for batch_idx in range(heatmap.size(0)):
            tmp_heatmap = np.zeros((x_size, y_size))
            sparse_indices = np.random.randint(0, total_indices, self.premeasured_num)
            """
            interval = total_indices // self.pre_measured_num  # Total number of indices divided by the number of sparse indices
            sparse_indices = [first_index + i * interval for i in range(self.pre_measured_num)]
            """
            if fixed_coord:
                if batch_idx == 0:
                    coordinates = []
                    for index in sparse_indices:
                        for dia_idx_x in range(patch_span):
                            for dia_idx_y in range(patch_span):
                                if x_size > (index // y_size) % x_size + dia_idx_x and y_size > (
                                        index % y_size) % y_size + dia_idx_y:
                                    coordinates.append(
                                        [(index // y_size) % x_size + dia_idx_x, (index % y_size) % y_size + dia_idx_y])

                    fixed_coords = coordinates
                else:
                    coordinates = fixed_coords
            else:
                coordinates = []
                for index in sparse_indices:
                    for dia_idx_x in range(patch_span):
                        for dia_idx_y in range(patch_span):
                            if x_size > (index // y_size) % x_size + dia_idx_x and y_size > (
                                    index % y_size) % y_size + dia_idx_y:
                                coordinates.append(
                                    [(index // y_size) % x_size + dia_idx_x, (index % y_size) % y_size + dia_idx_y])

                coordinates = np.array(coordinates, dtype=np.int64)

            for i, j in coordinates:
                if i < x_size and j < y_size:
                    adj_x_ind = int(round(i * x_ratio))
                    adj_y_ind = int(round(j * y_ratio))

                    # tmp_heatmap[i][j] = heatmap[batch_idx, 0, i, j]
                    tmp_heatmap[i][j] = med_rssi_real[batch_idx, 0, adj_x_ind, adj_y_ind]

            # tmp_heatmap = torch.from_numpy(tmp_heatmap)
            # tmp_heatmap = Image.fromarray(tmp_heatmap.astype('float32'), mode='RGB')
            import matplotlib.pyplot as plt
            cm = plt.get_cmap('plasma')
            tmp_heatmap = (255.0 * cm(tmp_heatmap)).astype('uint8')
            tmp_heatmap = Image.fromarray(tmp_heatmap[:, :, :3], 'RGB')
            # tmp_heatmap = TF.to_tensor(np.array(tmp_heatmap))
            # tmp_heatmap = torch.from_numpy(tmp_heatmap)

            # tmp_heatmap = Image.fromarray(tmp_heatmap[:, :, :3], 'RGB')
            # path_save_premeasured = '00_premeasured.png'
            # tmp_heatmap.save(path_save_premeasured)

            tmp_heatmap = TF.to_tensor(np.array(tmp_heatmap))
            # tmp_heatmap = TF.normalize(tmp_heatmap, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            coordinates = torch.from_numpy(np.array(coordinates))

            premeasured_heatmap.append(tmp_heatmap)
            premeasured_coords.append(coordinates)

        premeasured_heatmap = torch.stack(premeasured_heatmap, dim=0).cuda()
        premeasured_coords = torch.stack(premeasured_coords, dim=0).cuda()

        return premeasured_heatmap, premeasured_coords

    def get_premeasured_joint_image(self, heatmap, x_size, y_size, ap_coord):
        total_indices = x_size * y_size
        premeasured_heatmap = []
        premeasured_coords = []

        # patch_span = x_size // 20
        patch_span = 6
        if patch_span == 0:
            patch_span = 1
        for batch_idx in range(int(heatmap.size(0) // 2)):
            tmp_amp_heatmap = np.zeros((x_size, y_size))
            # tmp_phase_heatmap = np.full((x_size, y_size), fill_value=0)
            tmp_phase_heatmap = np.zeros((x_size, y_size))

            sparse_indices = np.random.randint(0, total_indices, self.premeasured_num)
            """
            interval = total_indices // self.pre_measured_num  # Total number of indices divided by the number of sparse indices
            sparse_indices = [first_index + i * interval for i in range(self.pre_measured_num)]
            """
            coordinates = []
            for index in sparse_indices:
                for dia_idx_x in range(patch_span):
                    for dia_idx_y in range(patch_span):
                        if x_size > (index // y_size) % x_size + dia_idx_x and y_size > (
                                index % y_size) % y_size + dia_idx_y:
                            coordinates.append(
                                [(index // y_size) % x_size + dia_idx_x, (index % y_size) % y_size + dia_idx_y])

            coordinates = np.array(coordinates, dtype=np.int64)
            for i, j in coordinates:
                if i < x_size and j < y_size:
                    tmp_amp_heatmap[i][j] = heatmap[batch_idx, 0, i, j]
                    tmp_phase_heatmap[i][j] = heatmap[batch_idx + 1, 0, i, j]

            # tmp_heatmap = torch.from_numpy(tmp_heatmap)
            # tmp_heatmap = Image.fromarray(tmp_heatmap.astype('float32'), mode='RGB')
            import matplotlib.pyplot as plt
            cm = plt.get_cmap('plasma')
            tmp_amp_heatmap = (255.0 * cm(tmp_amp_heatmap)).astype('uint8')
            tmp_amp_heatmap = Image.fromarray(tmp_amp_heatmap[:, :, :3], 'RGB')

            tmp_phase_heatmap = (255.0 * cm(tmp_phase_heatmap)).astype('uint8')
            tmp_phase_heatmap = Image.fromarray(tmp_phase_heatmap[:, :, :3], 'RGB')

            # tmp_heatmap = TF.to_tensor(np.array(tmp_heatmap))
            # tmp_heatmap = torch.from_numpy(tmp_heatmap)

            # tmp_heatmap = Image.fromarray(tmp_heatmap[:, :, :3], 'RGB')
            # path_save_premeasured = '00_premeasured.png'f
            # tmp_heatmap.save(path_save_premeasured)

            tmp_amp_heatmap = TF.to_tensor(np.array(tmp_amp_heatmap))
            tmp_phase_heatmap = TF.to_tensor(np.array(tmp_phase_heatmap))

            coordinates = torch.from_numpy(np.array(coordinates))

            premeasured_heatmap.append(tmp_amp_heatmap)
            premeasured_heatmap.append(tmp_phase_heatmap)

            premeasured_coords.append(coordinates)

        premeasured_heatmap = torch.stack(premeasured_heatmap, dim=0).cuda()
        premeasured_coords = torch.stack(premeasured_coords, dim=0).cuda()

        return premeasured_heatmap, premeasured_coords
    def normalize_data(self, data, target_range=(-150, -30)):
        """
        Normalize the given data to the specified target range.

        Parameters:
        - data: List or numpy array of data to be normalized.
        - target_range: Tuple representing the target range (default is (-90, -30)).


        Returns:
        - Normalized data.
        """
        # Convert data to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # data_array = data_array[:, :, np.logical_and(np.isfinite(data_array).all(axis=(1, 2, 3)), np.isfinite(data_array).all(axis=(1, 2, 3)))]
        # data_array[~np.isfinite(data_array)] = 0

        # Calculate the minimum and maximum of the data
        min_val = np.nanmin(data)
        max_val = np.nanmax(data) + 1e-8  # For blank handling
        # Normalize the data to the target range
        normalized_data = target_range[0] + ((data - min_val) / (max_val - min_val)) * (
                target_range[1] - target_range[0])

        # Use numpy.clip to set the minimum value
        # normalized_data = np.clip(normalized_data, real_min, None)
        # normalized_data[normalized_data == -57] = real_min

        return normalized_data
