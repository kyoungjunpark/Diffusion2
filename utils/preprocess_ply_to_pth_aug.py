import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import pandas as pd
import sys

FRONT_PATH = 'front_data_aug/output_ply_color_aug_30GHz'
OUT_DIR = 'front_data_aug/front_3d_processed_color_aug_30GHz'

def process_one_scene(fn):
    '''process one scene.'''
    if not os.path.exists(fn):
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    scene_name = fn.split('/')[-3]
    room_name = fn.split('/')[-2]
    name, extension = os.path.splitext(fn.split('/')[-1])
    object_name = name
    if os.path.exists(OUT_DIR + "/" + scene_name + "/" + room_name + '_'+object_name+'.pth'):
        print("exist: " + OUT_DIR + "/" + scene_name + "/" + room_name + '_'+object_name+'.pth')
        return
    os.makedirs(OUT_DIR + "/" + scene_name, exist_ok=True)
    try:
        a = plyfile.PlyData().read(fn)
    except Exception:
        return
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -4:-1])

    assert np.any(colors == [255, 255, 255]) and np.any(colors == [0, 0, 255])
    if np.min(coords) < -1000 or np.max(coords) > 1000:
        print("unexpected coords range")
        print(np.min(coords))
        print(np.max(coords))
        return
    if coords.shape[0] > 400000:
        print("too much coords: " + str(coords.shape[0]))
        return
    # no GT labels are provided, set all to 255
    labels = 255 * np.ones((coords.shape[0],), dtype=np.int32)
    print("Save: " + os.path.join(OUT_DIR + "/" + scene_name + "/" + room_name + '_'+object_name+'.pth'))
    torch.save((coords, colors, labels), os.path.join(OUT_DIR + "/" + scene_name + "/" + room_name + '_'+object_name+'.pth'))


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines


if __name__ == "__main__":
    scene_list = []
    for folder_name in os.listdir(FRONT_PATH):
        if os.path.isdir(FRONT_PATH + "/" + folder_name):
            for scene_name in os.listdir(FRONT_PATH + "/" + folder_name):
                if os.path.isdir(FRONT_PATH + "/" + folder_name + "/" + scene_name):
                    scene_list.append(FRONT_PATH + "/" + folder_name + "/" + scene_name + "/" + "meshes.ply")

    #####################################

    # category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)
    # mapping = np.insert(category_mapping[['nyu40id']].to_numpy().astype(int).flatten(), 0, 0, axis=0)
    files = []
    # for scene in scene_list:
    #     files = files+glob.glob(os.path.join(FRONT_PATH, scene, '*.ply'))
    def mute():
        sys.stdout = open(os.devnull, 'w')
    p = mp.Pool(processes=mp.cpu_count()*3)
    # print(scene_list)
    # print(len(scene_list_2))
    # exit(1)
    p.map(process_one_scene, scene_list)
    p.close()
    p.join()