"""
    This script generates a json file for the 3D Front dataset.
"""

import os
import argparse
import random
import json
import torch
import numpy as np

from utils.voxelizer import Voxelizer
from PIL import Image, ImageOps, ImageFile

parser = argparse.ArgumentParser(
    description="3D-Front jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the 3D Front Depth Completion dataset")

parser.add_argument('--path_human', type=str, required=True,
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

parser.add_argument('--video_fps', type=int, required=False,
                    default=10, help='fps')
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

FREQ_NAME = ['output_heatmap_aug_77GHz']


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def check_all_files_exist(heatmap_path_base, data_3d_path_base, house_name, room_name, video_fps):
    files_exist = True
    if not os.path.exists(os.path.join(data_3d_path_base, house_name, room_name, "room_meshes.pth")):
        return False
    for fps_idx in range(video_fps):
        if not os.path.exists(os.path.join(heatmap_path_base, house_name, room_name, str(fps_idx) + "_channel.png")):
            files_exist = False
            break
        if not os.path.exists(os.path.join(data_3d_path_base, house_name, room_name, str(fps_idx + 1) + "_meshes.pth")):
            files_exist = False
            break
    return files_exist



def generate_json():
    check_dir_existence(args.path_out)

    overshot_path_base = os.path.join(args.path_root, "output_with_AP_aug_image_final_0519")
    heatmap_path_list = []
    for freq_file_name in FREQ_NAME:
        heatmap_path_base = os.path.join(args.path_human, freq_file_name)
        heatmap_path_list.append(heatmap_path_base)

    data_3d_path_base = os.path.join(args.path_human, "front_3d_processed_color_aug")
    # dataset_list = os.listdir(path_base)
    video_fps = args.video_fps

    # For train/val splits

    dict_json = {}
    freq_toggle = 0

    for split in ['train', 'val', 'test']:
        # dataset_list.sort()
        dataset_list = os.listdir(heatmap_path_list[0])

        if split == 'train':
            dataset_list = dataset_list[:int(len(dataset_list) * 0.6)]
        elif split == "val":
            dataset_list = dataset_list[int(len(dataset_list) * 0.6):int(len(dataset_list) * 0.8)]
        elif split == "test":
            dataset_list = dataset_list[int(len(dataset_list) * 0.8):]
        else:
            raise AttributeError
        list_pairs = []
        for house_name in dataset_list:
            if house_name.endswith(".txt"):
                continue
            for room_name in os.listdir(heatmap_path_list[0] + '/' + house_name):
                is_correct_data = True
                room_path = os.path.join(data_3d_path_base, house_name, room_name, "room_meshes.pth")
                if not os.path.exists(room_path):
                    continue
                freq_toggle %= len(FREQ_NAME)

                # heatmap_path = os.path.join(heatmap_path_list[freq_toggle], house_name, room_name, "channel.png")

                locs_in, feats_in, labels_in = torch.load(room_path)
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

                if len(coords) > max_coords or not ap_True or not mesh_True or mesh_count > max_mesh_coords:
                    continue

                if not check_all_files_exist(heatmap_path_list[freq_toggle], data_3d_path_base, house_name, room_name, video_fps):
                    continue
                for fps_idx in range(video_fps):
                    overshot_path = os.path.join(overshot_path_base, house_name, room_name, str(fps_idx) + "_upper_shot.png")
                    heatmap_path = os.path.join(heatmap_path_list[freq_toggle], house_name, room_name, str(fps_idx) + "_channel.png")
                    data_3d_path = os.path.join(data_3d_path_base, house_name, room_name, str(fps_idx+1) + "_meshes.pth")

                    if not os.path.exists(overshot_path) or not os.path.exists(data_3d_path) or not os.path.exists(heatmap_path):
                        is_correct_data = False
                        break
                    if fps_idx == 0:
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

                        if damaged_image:
                            is_correct_data = False
                            break
                final_overshot_path = os.path.join(overshot_path_base, house_name, room_name, "upper_shot.png")
                final_heatmap_path = os.path.join(heatmap_path_list[freq_toggle], house_name, room_name)
                final_data_3d_path = os.path.join(data_3d_path_base, house_name, room_name)

                if is_correct_data:
                    dict_sample = {
                        'overshot': final_overshot_path,
                        'heatmap': final_heatmap_path,
                        'data_3d': final_data_3d_path,
                    }
                    print(house_name, room_name, freq_toggle)
                    list_pairs.append(dict_sample)
                else:
                    print("overshot: " + str(os.path.exists(final_overshot_path)))
                    print("data_3d_path: " + str(os.path.exists(final_heatmap_path)))
                    print("heatmap_path: " + str(os.path.exists(final_data_3d_path)))
                freq_toggle += 1

            # print("{} samples".format(house_name))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))
    # dict_json["train"].extend(dict_json["val"])
    # you can merge validation for better performance.
    print("{} split : Total {} samples".format("train", len(dict_json["train"])))

    random.shuffle(dict_json['train'])

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
