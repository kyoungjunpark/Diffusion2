"""
    This script generates a json file for the 3D Front dataset.
"""

import os
import argparse
import random
import json
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFile

from utils.voxelizer import Voxelizer
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="3D-Front jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the 3D Front Depth Completion dataset")

parser.add_argument('--path_human', type=str, required=False,
                    help="Path to the 3D Human dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='3d_front.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e12), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')
# python gen.py --path_root ~/front_data_aug_5GHz/ --path_out ../data_json_5GHz --name_out 3d_front_5GHz_1level.json

args = parser.parse_args()

random.seed(args.seed)
SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                      np.pi))
TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

ROTATION_AXIS = 'z'
LOCFEAT_IDX = 2
voxel_size = 0.02
voxelizer = Voxelizer(
    voxel_size=voxel_size,
    clip_bound=None,
    use_augmentation=True,
    scale_augmentation_bound=SCALE_AUGMENTATION_BOUND,
    rotation_augmentation_bound=ROTATION_AUGMENTATION_BOUND,
    translation_augmentation_ratio_bound=TRANSLATION_AUGMENTATION_RATIO_BOUND)


FREQ_NAME = ['output_heatmap_aug_5.16GHz', 'output_heatmap_aug_5.18GHz',
             'output_heatmap_aug_5.20GHz', 'output_heatmap_aug_5.22GHz', 'output_heatmap_aug_5.24GHz',
             'output_heatmap_aug_5.26GHz', 'output_heatmap_aug_5.28GHz', 'output_heatmap_aug_5.30GHz',
             'output_heatmap_aug_5.32GHz']

# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def generate_json():
    check_dir_existence(args.path_out)

    overshot_path_base = os.path.join(args.path_root, "output_with_AP_aug_image")
    heatmap_path_list = []
    for freq_file_name in FREQ_NAME:
        heatmap_path_base = os.path.join(args.path_root, freq_file_name)
        heatmap_path_list.append(heatmap_path_base)

    data_3d_path_base = os.path.join(args.path_root, "front_3d_processed_color_aug")
    # dataset_list = os.listdir(path_base)

    # For train/val splits
    # freq_toggle = 0

    dict_json = {}
    list_pairs = []
    count = 0
    freq_toggle = 0

    dataset_list = os.listdir(heatmap_path_list[0])
    for house_name in tqdm(dataset_list):
        if house_name.endswith(".txt") or not os.path.isdir(heatmap_path_list[0] + '/' + house_name):
            continue
        for room_name in os.listdir(heatmap_path_list[0] + '/' + house_name):
            overshot_path = os.path.join(overshot_path_base, house_name, room_name, "upper_shot.png")
            data_3d_path = os.path.join(data_3d_path_base, house_name, room_name + "_meshes.pth")
            freq_toggle %= len(FREQ_NAME)
            heatmap_path = os.path.join(heatmap_path_list[freq_toggle], house_name, room_name, "channel.png")

            if os.path.exists(overshot_path) and os.path.exists(data_3d_path) and os.path.exists(heatmap_path):
                # for heatmap_path_tmp in heatmap_path_list:
                #     assert os.path.exists(os.path.join(heatmap_path_tmp, house_name, room_name, "phase.png")), (heatmap_path_tmp, house_name, room_name)

                locs_in, feats_in, labels_in = torch.load(data_3d_path)
                locs, feats, labels, inds_reconstruct = voxelizer.voxelize(
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
                ap_True = mesh_True = False
                for idx in range(len(feats)):
                    if torch.equal(feats[idx], ap_torch):
                        ap_True = True
                        break

                tmp_feats = torch.tensor([0., 0., 255.])
                for idx in range(len(feats)):
                    if torch.equal(feats[idx], tmp_feats):
                        mesh_True = True
                        mesh_count += 1

                def is_image_all_black_or_white(image_path):
                    img = Image.open(image_path)
                    grayscale_image = img.convert('L')
                    pixels = grayscale_image.load()
                    # Iterate over each pixel
                    for y in range(grayscale_image.size[1]):
                        for x in range(grayscale_image.size[0]):
                            # Check if the pixel is not completely black (0) or white (255)
                            if pixels[x, y] != 0 and pixels[x, y] != 255:
                                return False

                    # If all pixels are either black or white, return True
                    return True

                try:
                    heatmap = Image.open(heatmap_path)
                    damaged_image = False

                    heatmap.resize((352, 706))
                    all_black_or_white = is_image_all_black_or_white(heatmap_path)
                    if all_black_or_white:
                        damaged_image = True
                        print("all black or white")
                except OSError:
                    damaged_image = True
                if len(coords) > max_coords or not ap_True or not mesh_True or mesh_count > max_mesh_coords or damaged_image:
                    continue

                dict_sample = {
                    'overshot': overshot_path,
                    'heatmap': heatmap_path,
                    'data_3d': data_3d_path,
                }
                print(dict_sample)
                list_pairs.append(dict_sample)
            else:
                print("Some files do not exist")

            freq_toggle += 1
        count += 1

    # random.shuffle(list_pairs)
    dict_json["train"] = list_pairs[:int(len(list_pairs) * 0.6)]
    dict_json["test"] = list_pairs[int(len(list_pairs) * 0.6):int(len(list_pairs) * 0.8)]
    dict_json["val"] = list_pairs[int(len(list_pairs) * 0.8):]

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val),
              ('test', args.num_test)]:
        if len(dict_json[s[0]]) > s[1]:
            # Do shuffle
            random.shuffle(dict_json[s[0]])

            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig,
                                               len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('')

    generate_json()
