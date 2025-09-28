import os
import numpy as np
import json
import random

from utils.voxelizer import Voxelizer
from . import BaseDataset
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from model.ops.depth_map_proc import simple_depth_completion
from torchvision.transforms import InterpolationMode
import logging
import utils.augmentation as t

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

"""
3D Front json file has a following format:-

{
    "train": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }, ...
    ],
    "val": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }, ...
    ],
    "test": [
        {
            "overshot": "output_with_AP_image/House/LivingRoom/over_shot.png",
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }, ...
    ]
}
"""


class Front3D_1D_Video(BaseDataset):
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, args, mode):
        super(Front3D_1D_Video, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width
        # self.height = 228
        # self.width = 304
        self.video_fps = args.video_fps

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

        # Random shuffle
        random.shuffle(self.sample_list)

        self.voxel_size = 0.02

        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND)

        data_aug_color_trans_ratio = 0.1
        data_aug_color_jitter_std = 0.05
        data_aug_hue_max = 0.5
        data_aug_saturation_max = 0.2
        prevoxel_transform_train = [
            t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
        self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
        input_transforms = [
            t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
            # t.ChromaticAutoContrast(),
            # t.ChromaticTranslation(data_aug_color_trans_ratio),
            # t.ChromaticJitter(data_aug_color_jitter_std),
            # t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
        ]
        self.input_transforms = t.Compose(input_transforms)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        overshot, heatmap, data_3d, freq_level, room_info = self._load_data(idx)
        overshots = []
        heatmaps = []
        for frame_idx in range(self.video_fps):
            overshot_tmp = TF.to_tensor(overshot[frame_idx])
            overshot_tmp = TF.normalize(overshot_tmp, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
            # overshot = overshot.to(torch.float16)
            overshots.append(overshot_tmp)

            heatmap_tmp = TF.to_tensor(heatmap[frame_idx])
            # heatmap = heatmap.to(torch.float16)
            heatmaps.append(heatmap_tmp)
            assert not torch.isnan(overshot_tmp).any(), overshot_tmp
            assert not torch.isnan(heatmap_tmp).any(), heatmap_tmp

        overshots = torch.stack(overshots)
        heatmaps = torch.stack(heatmaps)

        output = {'overshot': overshots, 'heatmap': heatmaps, 'freq_level': freq_level,
                  'data_3d': data_3d}

        return output

    def _load_data(self, idx):
        overshot_path = os.path.join(self.args.dir_data,
                                     self.sample_list[idx]['overshot'])
        heatmap_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['heatmap'])
        data_3d_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['data_3d'])
        overshot_path = "/".join(overshot_path.split("/")[:-1])
        room_3d_path = os.path.join(data_3d_path, "room_meshes.pth")


        import re
        def extract_digits_between(text):
            pattern = r'_(\d+(\.\d+)?)GHz'
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                return None

        result = extract_digits_between(heatmap_path)
        freq_level = torch.tensor(float(result)).float()

        heatmap_list = []
        overshot_list = []
        data_3d_list = []
        room_info_list = []

        freq_level_list = [freq_level] * self.video_fps
        _scale = np.random.uniform(1.0, 1.5)

        locs_in_r, feats_in_r, labels_in_r = torch.load(room_3d_path)

        for frame_idx in range(self.video_fps):
            heatmap = Image.open(os.path.join(heatmap_path, str(frame_idx) + "_channel.png"))

            overshot = Image.open(os.path.join(overshot_path, str(frame_idx) + "_upper_shot.png"))
            overshot = overshot.resize((self.width, self.height))
            overshot = overshot.convert('RGB')

            try:
                heatmap = heatmap.resize((self.width, self.height))
                heatmap = heatmap.convert('L')
                heatmap = np.array(heatmap)
            except OSError:
                print(heatmap_path)
                exit(1)

            heatmap = heatmap.astype(np.float32) / 256.0
            heatmap = Image.fromarray(heatmap.astype('float32'), mode='F')

            locs_in_h, feats_in_h, labels_in_h = torch.load(os.path.join(data_3d_path, str(frame_idx+1) + "_meshes.pth"))

            locs_in = np.concatenate((locs_in_h, locs_in_r), axis=0)
            feats_in = np.concatenate((feats_in_h, feats_in_r), axis=0)
            labels_in = np.concatenate((labels_in_h, labels_in_r), axis=0)

            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            # no color in the input point cloud, e.g nuscenes
            # assert not np.isscalar(feats_in) and feats_in != 0
            # np.set_printoptions(suppress=True)
            # locs_in = np.round(locs_in, 6)
            # locs_in[locs_in == -0.] = 0.

            # f.write("\n".join(str(item) for item in locs))
            locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in)

            # assert np.any(feats == [255., 255., 255.]), (f.write("\n".join(str(item) for item in feats)), data_3d_path)
            # assert np.any(feats == [0., 0., 255.]), (f.write("\n".join(str(item) for item in feats)), data_3d_path)

            coords = torch.from_numpy(locs).int()
            # Add batch info to the coords
            coords = torch.cat(
                (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)

            # Batch considering
            coords[:, 0] *= (frame_idx + 1)
            # coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

            feats = torch.from_numpy(feats).float()
            labels = torch.from_numpy(labels).long()
            max_coords = 30000
            max_mesh_coords = 3000
            # Random sampling (due to CUDA mem issue)
            # if torch.all(torch.eq(feat, torch.tensor([255., 255., 255.]).to('cuda'))):
            ap_torch = torch.tensor([255., 255., 255.])
            mesh_count = 0
            ap_tmp_coords = []
            ap_tmp_feats = []
            mesh_tmp_coords = []
            mesh_tmp_feats = []
            ap_True = mesh_True = False
            for idx in range(len(feats)):
                if torch.equal(feats[idx], ap_torch):
                    ap_True = True
                    ap_tmp_coords.append(coords[idx])
                    ap_tmp_feats.append(feats[idx])

            tmp_feats = torch.tensor([0., 0., 255.])
            for idx in range(len(feats)):
                if torch.equal(feats[idx], tmp_feats):
                    mesh_True = True
                    mesh_count += 1
                    mesh_tmp_coords.append(coords[idx])
                    mesh_tmp_feats.append(feats[idx])

            assert ap_True and mesh_True

            # assert is_True, (f.write("\n".join(str(item) for item in feats_in)), f2.write("\n".join(str(item) for item in locs_in)), data_3d_path)
            # assert (any(torch.equal(feat, tmp_feats) for feat in feats)), (f.write("\n".join(str(item) for item in feats.cpu().detach().numpy())), data_3d_path)
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords).int()
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats).float()
            assert coords.size(0) == feats.size(0)
            # feats = (feats / 127.5) -ioc 1

            data_3d = (coords, feats, labels)

            w1, h1 = overshot.size
            w2, h2 = heatmap.size
            assert w1 == w2 and h1 == h2, print(w1, w2, h1, h2)

            overshot_list.append(overshot)
            heatmap_list.append(heatmap)
            data_3d_list.append(data_3d)
            room_info_list.append(overshot_path)
        # data_3d_list = list(zip(*data_3d_list))

        return overshot_list, heatmap_list, data_3d_list, freq_level_list, room_info_list
