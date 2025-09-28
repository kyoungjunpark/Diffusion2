import os
import numpy as np
import json

from utils.voxelizer import Voxelizer
from . import BaseDataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import logging
import utils.augmentation as t

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

"""
3D Front json file has a following format:-

{
    "train": [
        {
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "coords": "output_heatmap/House/LivingRoom/coords.npy",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }, ...
    ],
    "val": [
        {
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "coords": "output_heatmap/House/LivingRoom/coords.npy",
            "front3d": "output_heatmap/House/LivingRoom-162963_meshes.pth"
        }, ...
    ],
    "test": [
        {
            "heatmap": "output_heatmap/House/LivingRoom/channel.png",
            "coords": "output_heatmap/House/LivingRoom/coords.npy",
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


class MRI(BaseDataset):
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2

    def __init__(self, args, mode):
        super(MRI, self).__init__(args, mode)

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
        output = {'overshot': overshot, 'heatmap': heatmap, 'data_3d': data_3d}

        return output

    def _load_data(self, idx):
        heatmap_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['heatmap'])
        coords_path = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['heatmap'].split("/")[:-1] + "/coords.npy")

        # heatmap = read_depth(heatmap_path)
        heatmap = Image.open(heatmap_path)

        heatmap = heatmap.resize((self.width, self.height))

        heatmap = heatmap.convert('L')

        # heatmap = ImageOps.grayscale(heatmap)

        heatmap = np.array(heatmap)
        coords= np.load(coords_path)

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
        # heatmap = Image.fromarray(heatmap.astype('float32'), mode='F')
        # heatmap = np.uint8(heatmap*255)
        heatmap = Image.fromarray(heatmap.astype('float32'), mode='F')

        return heatmap, coords
