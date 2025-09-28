import os
import numpy as np
import json
import random

from utils.voxelizer import Voxelizer
from . import BaseDataset
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
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


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))
    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


class Front3D_1D(BaseDataset):
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, args, mode):
        super(Front3D_1D, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width
        # self.height = 228
        # self.width = 304

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

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
        overshot, heatmap, data_3d = self._load_data(idx)

        if self.mode in ['train', 'val']:
            # Top crop if needed
            overshot = TF.to_tensor(overshot)
            overshot = TF.normalize(overshot, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            heatmap = TF.to_tensor(np.array(heatmap))

        else:
            """
            if self.args.top_crop > 0 and self.args.test_crop:
                width, height = overshot.size
                overshot = TF.crop(overshot, self.args.top_crop, 0, height - self.args.top_crop, width)
                heatmap = TF.crop(heatmap, self.args.top_crop, 0, height - self.args.top_crop, width)
            """

            overshot = TF.to_tensor(overshot)
            overshot = TF.normalize(overshot, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

            heatmap = TF.to_tensor(np.array(heatmap))
        transform = T.Compose([
            T.PILToTensor()
        ])
        # heatmap = transform(heatmap).float()
        # heatmap_a = transform(heatmap_a)

        """
        print("load")
        print(heatmap.size())
        print(heatmap)
        print(heatmap_a.size())
        print(heatmap_a)
        """
        assert not torch.isnan(overshot).any(), overshot
        assert not torch.isnan(heatmap).any(), overshot

        output = {'overshot': overshot, 'heatmap': heatmap, 'data_3d': data_3d}

        return output

    def _load_data(self, idx):
        overshot_path = os.path.join(self.args.dir_data,
                                     self.sample_list[idx]['overshot'])
        heatmap_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['heatmap'])
        data_3d_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['data_3d'])

        # heatmap = read_depth(heatmap_path)
        heatmap = Image.open(heatmap_path)

        # heatmap = read_depth(heatmap_path)
        overshot = Image.open(overshot_path)
        # heatmap = Image.fromarray(heatmap.astype('float32'), mode='RGB')

        # data_3d = torch.load(data_3d_path)
        # logging.debug("Rotate 90 degree of overshot image to sync with the heatmap")
        # overshot = overshot.rotate(270)
        """
        t_func = T.Compose([
                T.Resize(self.height),
                T.ToTensor()])
        overshot = t_func(overshot)
        heatmap = t_func(heatmap)
        """
        _scale = np.random.uniform(1.0, 1.5)
        overshot = overshot.resize((self.width, self.height))
        heatmap = heatmap.resize((self.width, self.height))
        overshot = overshot.convert('RGB')

        heatmap = heatmap.convert('L')

        # heatmap = ImageOps.grayscale(heatmap)

        heatmap = np.array(heatmap)
        from skimage.color import rgb2hsv
        # cm = plt.get_cmap('jet')
        # [~, idx] = sortrows(rgb2hsv(cm), -1);
        # heatmap = rgb2hsv(heatmap[:,:,:3])
        # cm = plt.get_cmap('jet')
        # heatmap = cm(heatmap)
        # heatmap = rgb_to_hsv(heatmap[:, :, :3])  # Extract the RGB channels and convert to HSV
        # print(np.array(heatmap))
        # heatmap = heatmap[:, :, 0]  # Use the Hue channel as scalar values
        # heatmap = np.dot(heatmap[:, :, :3], [0.299, 0.587, 0.114])
        heatmap = heatmap.astype(np.float32) / 256.0
        heatmap = Image.fromarray(heatmap.astype('float32'), mode='F')
        # heatmap = np.uint8(heatmap*255)
        # heatmap = Image.fromarray(heatmap.astype('float32'), mode='F')

        # heatmap = heatmap.convert('P')
        # heatmap = heatmap.convert('L')
        # heatmap = heatmap.convert('RGB')
        """
        print("import")
        print(np.array(heatmap))
        print(np.array(heatmap_a))
        """

        locs_in, feats_in, labels_in = torch.load(data_3d_path)

        labels_in[labels_in == -100] = 255
        labels_in = labels_in.astype(np.uint8)
        # no color in the input point cloud, e.g nuscenes
        # assert not np.isscalar(feats_in) and feats_in != 0
        """
        if np.isscalar(feats_in) and feats_in == 0:
            feats_in = np.zeros_like(locs_in)
        """
        # feats_in = (feats_in + 1.) * 127.5
        """
        prevoxel_transform_train = [
            t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)]
        self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
        # locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        labels = labels_in

        locs, feats, labels = self.input_transforms(locs, feats, labels)

        feats = torch.from_numpy(feats).float() / 127.5 - 1.
        # feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()
        """
        # np.set_printoptions(suppress=True)
        # locs_in = np.round(locs_in, 6)
        # locs_in[locs_in == -0.] = 0.

        f = open('error_feats', 'w')
        f2 = open('error_coords', 'w')

        # f.write("\n".join(str(item) for item in locs))
        locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
            locs_in, feats_in, labels_in)

        # assert np.any(feats == [255., 255., 255.]), (f.write("\n".join(str(item) for item in feats)), data_3d_path)
        # assert np.any(feats == [0., 0., 255.]), (f.write("\n".join(str(item) for item in feats)), data_3d_path)

        coords = torch.from_numpy(locs).int()
        # Add batch info to the coords
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
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
        assert ap_True, (
            f.write("\n".join(str(item) for item in feats_in)), f2.write("\n".join(str(item) for item in locs_in)))
        if not ap_True:
            print("Should not happen2")
            raise Exception
        if not mesh_True:
            print("Should not happen3")
            raise Exception

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
        return overshot, heatmap, data_3d

