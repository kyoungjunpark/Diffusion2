import os
import argparse
import random
import json
from pathlib import Path
import multiprocessing as mp
import numpy as np

parser = argparse.ArgumentParser(
    description="3D-Front jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the 3D Front Depth Completion dataset")

args = parser.parse_args()


def process_one_scene(house_name):
    heatmap_path_base = os.path.join(args.path_root, "output_heatmap_rgb")
    heatmap_output_path_base = os.path.join(args.path_root, "output_heatmap_gray")
    if house_name.endswith(".txt"):
        return
    for room_name in os.listdir(heatmap_path_base + '/' + house_name):
        heatmap_path = os.path.join(heatmap_path_base, house_name, room_name, "channel.png")
        output_path = os.path.join(heatmap_output_path_base, house_name, room_name, "channel.png")
        Path(os.path.join(heatmap_output_path_base, house_name, room_name)).mkdir(parents=True, exist_ok=True)

        if os.path.exists(os.path.join(heatmap_output_path_base, house_name, room_name, "channel.png")):
            continue
        os.system("python colormap_to_grayscale.py input_cmap.png {} -o {}".format(heatmap_path, output_path))

    print("{} samples".format(house_name))


heatmap_path_base = os.path.join(args.path_root, "output_heatmap_rgb")
dataset_list = os.listdir(heatmap_path_base)
for sub_directory in np.array_split(dataset_list, 100):
    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_one_scene, sub_directory)
    p.close()
    p.join()
