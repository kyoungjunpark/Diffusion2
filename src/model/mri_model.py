# Copyright (c) Phigent Robotics. All rights reserved.

"""
# from model.builder import DETECTORS
# from model.builder import HEADS, build_loss
from model.backbone import get as get_backbone
from model.head import get as get_head
"""
import os
import numpy as np
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# It is needed for head registration
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as ssim

from model.utils import normalize_data
import shutil
from baseline_utils.logger import logger_config


class MRI_Model():
    def __init__(self, args):
        super(MRI_Model, self).__init__()
        self.args = args
        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data['test']
        self.results = []
        self.coords = []
        self.AP = []
        self.heatmap_path = []

        self.load_dataset()
        self.execute()

    def load_dataset(self):
        """
        for idx in range(len(self.json_data['test'])):
            heatmap_path = os.path.join(self.args.dir_data,
                                        self.json_data['test'][idx]['heatmap'])
            coords_path = os.path.join(self.args.dir_data,
                                        "/".join(self.json_data['test'][idx]['heatmap'].split("/")[:-1]) + "/coords.npy")
        """
        for idx in range(len(self.sample_list)):
            heatmap_path = os.path.join(self.args.dir_data,
                                        self.sample_list[idx]['heatmap'])
            sub_path = "/".join(heatmap_path.split("/")[:-1])
            # heatmap = read_depth(heatmap_path)
            if os.path.exists(sub_path + "/results.npy") and os.path.exists(sub_path + "/coords.npy") and os.path.exists(sub_path + "/AP.npy"):

                results = np.load(sub_path + "/results.npy")
                coords = np.load(sub_path + "/coords.npy")
                AP = np.load(sub_path + "/AP.npy")

                self.results.append(results)
                self.coords.append(coords)
                self.AP.append(AP)
                self.heatmap_path.append(heatmap_path)

                # break

    def execute(self):
        # print(self.heatmap)
        # print(self.coords)
        # exit(1)
        patch_span = 3
        pre_measured_num = 15 * patch_span * patch_span
        coordinates_list = []
        distance_list = []
        errors_list = []
        errors_list2 = []
        mean_ssim_list = []
        idx = 0
        for results, coords, AP, heatmap_path in zip(self.results, self.coords, self.AP, self.heatmap_path):
            coordinates = []

            results = np.clip(results, a_min=None, a_max=-30)
            results = normalize_data(results, target_range=(-57, -30))

            predicted_result = np.full((results.shape[0], results.shape[1]), -57)
            actual_points = int(pre_measured_num // patch_span // patch_span)

            total_indices = results.shape[0] * results.shape[1]
            first_index = np.random.randint(0, total_indices)
            interval = total_indices // pre_measured_num  # Total number of indices divided by the number of sparse indices
            sparse_indices = [first_index + i * interval for i in range(actual_points)]
            # Pixel-specific measurement
            for index in sparse_indices:
                for dia_idx_x in range(patch_span):
                    for dia_idx_y in range(patch_span):
                        x_index = min(results.shape[0] - 1, (index // results.shape[1]) % results.shape[0] + dia_idx_x)
                        y_index = min(results.shape[1] - 1, (index % results.shape[1]) % results.shape[1] + dia_idx_y)
                        coordinates.append([x_index, y_index])

            dis = [np.linalg.norm(AP - coords[coordinate[0]][coordinate[1]]) for coordinate in coordinates]
            rssi = [results[coordinate[0]][coordinate[1]] for coordinate in coordinates]
            # coordinates_list.append(coordinates)
            # distance_list.append(dis)

            errors = []
            results_tmp = []
            rssi_pred_tmp = []

            T, gamma = fit_T_gamma(dis, rssi)
            for i in range(coords.shape[0]):
                for j in range(coords.shape[1]):
                    if (i, j) in coordinates:
                        predicted_result[i][j] = results[i][j]
                        continue
                    dis = np.linalg.norm(AP - coords[i][j])
                    rssi_pred = T - 10 * gamma * np.log10(dis)
                    if rssi_pred < -57:
                        rssi_pred = -57

                    predicted_result[i][j] = rssi_pred
                    # print(rssi, rssi_pred)
                    error = abs(results[i][j] - rssi_pred)
                    results_tmp.append(results[i][j])
                    rssi_pred_tmp.append(rssi_pred)
                    errors.append(error)
            # os.makedirs(os.path.join(self.args.save_dir, str(idx)))

            # save_heatmap(predicted_result, os.path.join(self.args.save_dir, str(idx), 'predicted_RSSI.png'))
            # save_heatmap(results, os.path.join(self.args.save_dir, str(idx), 'gt.png'))
            try:
                ssim_value = ssim(np.array(results_tmp), np.array(rssi_pred_tmp), data_range=70)
            except ValueError:
                print("ssim error")
                ssim_value = 1
            # shutil.copy(heatmap_path, self.args.save_dir)
            """
            with open(os.path.join(self.args.save_dir, str(idx), "room_name.txt"), 'w') as f:
                f.write(heatmap_path)
                f.write(str(np.mean(errors)))
                f.write(str(np.median(errors)))
            """
            errors_list.append(np.mean(errors))
            errors_list2.append(np.median(errors))
            mean_ssim_list.append(ssim_value)
            print("Stacked Mean error:%.2f".format(np.mean(errors_list)))
            print("Stacked Median error:%.2f".format(np.median(errors_list)))
            print("Stacked SSIM:%.2f".format(np.mean(mean_ssim_list)))

            idx += 1
        print("test mean error: {:.2f} dB".format(np.mean(errors_list)))
        print("test median error: {:.2f} dB".format(np.median(errors_list2)))
        np.save(os.path.join(self.args.save_dir, "error_elems.npy"), np.array(errors_list))
        np.save(os.path.join(self.args.save_dir, "error_ssim_elems.npy"), np.array(mean_ssim_list))

        # print(f"fitting T: {T}, gamma: {gamma}")

        # print(coordinates_list)
        # print(first_index)
        # print(interval)

        return None


def fit_T_gamma(distance, RSSI):
    """RSSI = T - 10 * gamma * log10(d)
    """

    log_distance = np.log10(np.array(distance)).reshape(-1, 1)
    RSSI = np.array(RSSI).reshape(-1, 1)
    model = LinearRegression()
    model.fit(log_distance, RSSI)

    # The slope (m) and intercept (c) of the model:
    m = model.coef_[0][0]
    c = model.intercept_[0]
    T = c
    gamma = -m / 10

    return T, gamma


def save_heatmap(result, file_name):
    plt.imshow(result, norm=None, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    plt.axis('off')

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

    original_image = Image.open(file_name)

    # Set the desired resolution
    new_width = 1920
    new_height = 1080

    # Resize the image
    resized_image = original_image.resize((new_width, new_height))

    # Save the resized image
    resized_image.save(file_name)