

import torch

from . import BaseMetric
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math
from skimage.metrics import structural_similarity as ssim
import pickle

cm = plt.get_cmap('gray')

diffusion_min = -120
real_min = -57


def normalize_data(data, target_range=(-150, -30)):
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


def get_mean(x_index, y_index,data):
    n = 2.5
    total_points = 0
    total_rssi = 0
    for i in range(0, 40):
        for j in range(0, 65):
            # 欧几里得距离
            if (i - x_index) ** 2 + (j - y_index) ** 2 <= n**2:
                total_points += 1
                total_rssi += data[i][j]
    return total_rssi / total_points


def calculate_error(real_rssi, sim_data, x_ratio, y_ratio):
    real_data = []
    rounded_sim_data = []
    for i, point in enumerate(real_rssi):
        point["med_rssi"] = max(point["med_rssi"], -70)
        real_data.append(point["med_rssi"])
        # x_index = round((point["x"] + 4) * 5)-10
        # y_index = round((point["y"] + 6.5) * 5)
        print(point, x_ratio, real_data)
        print(sim_data.shape)
        x_index = round((point["x"] * x_ratio))
        y_index = round((point["y"] * y_ratio))
        print(x_index, y_index)
        print("==")

        # data = sim_data[x_index][y_index]
        # data 取 x_index, y_index 附近的均值
        data = get_mean(x_index, y_index, sim_data)
        rounded_sim_data.append(data)
    return np.mean(np.abs(np.array(real_data) - np.array(rounded_sim_data)))


def find_closest_coordinate(x, y, target_matrix, z=0):
    # Flatten the matrix to simplify the distance calculation
    flattened_matrix = target_matrix.reshape(-1, 3)

    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((flattened_matrix - np.array([x, y, z])) ** 2, axis=1))

    # Find the index of the minimum distance
    closest_index = np.argmin(distances)

    # Convert the flattened index back to matrix indices
    closest_coordinate = np.unravel_index(closest_index, target_matrix.shape[:-1])

    return closest_coordinate, closest_index


def find_matching_coordinates(x, y, target_matrix, z=0, threshold=0.05):
    # Flatten the matrix to simplify the distance calculation
    flattened_matrix = target_matrix.reshape(-1, 3)
    coord_list = []
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((flattened_matrix[:, :2] - np.array([x, y])) ** 2, axis=1))

    # Find the index of the minimum distance
    close_indices = np.where(distances <= threshold)[0]

    if len(close_indices) == 0:
        return None
    for close_index in close_indices:
        coord_list.append(flattened_matrix[close_index])
    # Convert the flattened index back to matrix indices
    closest_coordinate = np.unravel_index(close_indices, target_matrix.shape[:-1])
    return closest_coordinate


def reshape_with_average_pooling(arr, new_width, new_height):
    original_width, original_height = arr.shape

    if new_width > original_width or new_height > original_height:
        raise ValueError("New width and height must be smaller than or equal to the original dimensions.")

    # Calculate the block size in each dimension
    block_size_x = original_width // new_width
    block_size_y = original_height // new_height

    # Reshape the array by averaging each block
    reshaped_arr = np.zeros((new_width, new_height), dtype=arr.dtype)
    for i in range(new_width):
        for j in range(new_height):
            start_x = i * block_size_x
            end_x = start_x + block_size_x
            start_y = j * block_size_y
            end_y = start_y + block_size_y
            block = arr[start_x:end_x, start_y:end_y]
            reshaped_arr[i, j] = np.mean(block)

    return reshaped_arr


def remove_outliers(data, percentage=5):
    # Calculate the lower and upper bounds
    data = np.array(data)
    lower_bound = np.percentile(data, percentage)
    upper_bound = np.percentile(data, 100 - percentage)

    # Mask the values within the bounds
    mask = (data >= lower_bound) & (data <= upper_bound)

    # Return the filtered array
    return data[mask]


class Diffusion_Heatmap_Metric(BaseMetric):
    def __init__(self, args):
        super(Diffusion_Heatmap_Metric, self).__init__(args)

        self.args = args
        # self.t_valid = 0.0001
        if "A_P_Joint" in args.head_specify:
            self.amp_metric_name = [
                'dB(mean)', 'dB(median)', 'SSIM(mean)', 'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
            ]
            self.phase_metric_name = [
                'rad(mean)', 'rad(median)', 'SSIM(mean)', 'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
            ]
            self.metric_name = self.amp_metric_name + self.phase_metric_name
        elif "Phase" in args.head_specify:
            self.metric_name = [
                'rad(mean)', 'rad(median)', 'pathlen(mean)', 'pathlen(median)', 'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
            ]
        else:
            self.metric_name = [
                'dB(mean)', 'dB(median)', 'SSIM(mean)', 'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
            ]

    def evaluate(self, sample, output, mode):
        # Handle joint evaluate
        if "A_P_Joint" in self.args.head_specify:
            return self.evaluate_joint(sample, output, mode)
        elif "Phase" in self.args.head_specify:
            return self.evaluate_phase(sample, output, mode)

        with torch.no_grad():
            pred = output['pred'].detach()
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')

            heatmap = sample['heatmap'].detach()
            # heatmap = heatmap[0, 0, :, :].data.cpu().numpy()
            # heatmap = (255.0 * cm(heatmap)).astype('uint8')
            assert not torch.isnan(pred).any().item(), pred
            assert not torch.isnan(heatmap).any().item(), heatmap

            pred_inv = 1.0 / (pred + 1e-8)
            heatmap_inv = 1.0 / (heatmap + 1e-8)

            # For numerical stability
            # mask = heatmap > self.t_valid
            # num_valid = mask.sum()

            # pred = pred[mask]
            # heatmap = heatmap[mask]

            # pred_inv = pred_inv[mask]
            # heatmap_inv = heatmap_inv[mask]

            # pred_inv[pred <= self.t_valid] = 0.0
            # heatmap_inv[heatmap <= self.t_valid] = 0.0
            num_valid = torch.numel(pred)
            # RMSE / MAE / dB
            pred_norm_np = normalize_data(pred.cpu(), target_range=(-57, -30))
            heatmap_norm_np = normalize_data(heatmap.cpu(), target_range=(-57, -30))

            pred_norm = torch.from_numpy(pred_norm_np).to('cuda')
            heatmap_norm = torch.from_numpy(heatmap_norm_np).to('cuda')
            assert not torch.isnan(pred_norm).any().item(), pred_norm
            assert not torch.isnan(heatmap_norm).any().item(), heatmap_norm

            diff_norm = pred_norm - heatmap_norm
            diff_abs_norm = torch.abs(diff_norm)

            diff = pred - heatmap
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)
            # 100?
            dB_mean = torch.nanmean(diff_abs_norm)
            dB_median = torch.nanmedian(diff_abs_norm)

            # SSIM
            # pred_norm_np - heatmap_norm_np
            try:
                ssim_value = torch.tensor(ssim(pred_norm_np[0][0], heatmap_norm_np[0][0], data_range=70), device='cuda')
            except ValueError as e:
                ssim_value = torch.tensor(1, device='cuda')

            # iRMSE / iMAE
            diff_inv = pred_inv - heatmap_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (heatmap + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = heatmap / (pred + 1e-8)
            r2 = pred / (heatmap + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [dB_mean, dB_median, ssim_value, rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()
        return result

    def evaluate_phase(self, sample, output, mode):
        with torch.no_grad():
            pred = output['pred'].detach()
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')

            freq = sample['freq_level'].detach()  # 77
            freq *= 10**9
            wave_len = 3e8 / freq
            wave_len = wave_len.view(freq.size(0), 1, 1, 1)

            heatmap = sample['heatmap'].detach()
            # heatmap = heatmap[0, 0, :, :].data.cpu().numpy()
            # heatmap = (255.0 * cm(heatmap)).astype('uint8')
            assert not torch.isnan(pred).any().item(), pred
            assert not torch.isnan(heatmap).any().item(), heatmap

            # Do normalize into origianl pathlength range(1-25) from RGB(0-255)
            phase_pred_norm = torch.from_numpy(normalize_data(pred.cpu(), target_range=(1, 25))).to('cuda')
            phase_heatmap_norm = torch.from_numpy(normalize_data(heatmap.cpu(), target_range=(1, 25))).to('cuda')

            length_diff_norm = phase_pred_norm - phase_heatmap_norm
            length_diff_abs_norm = torch.abs(length_diff_norm)
            len_mean = torch.nanmean(length_diff_abs_norm)
            len_median = torch.nanmedian(length_diff_abs_norm)

            assert not torch.isnan(phase_pred_norm).any().item(), phase_pred_norm
            assert not torch.isnan(phase_heatmap_norm).any().item(), phase_heatmap_norm
            # =. 2 pi D/lambda , lambda = 3e8/77e9

            # pathlength into phase
            phase_pred_norm = 2 * math.pi * phase_pred_norm / wave_len
            phase_heatmap_norm = 2 * math.pi * phase_heatmap_norm / wave_len

            phase_mask = (phase_heatmap_norm != float('inf')) & (phase_heatmap_norm != -float('inf'))
            phase_pred_norm = phase_pred_norm[phase_mask]
            phase_heatmap_norm = phase_heatmap_norm[phase_mask]

            # unwrap into wrap (0-2pi)
            phase_pred_norm = phase_pred_norm % (2 * math.pi)
            phase_heatmap_norm = phase_heatmap_norm % (2 * math.pi)

            phase_pred_inv = 1.0 / (phase_pred_norm + 1e-8)
            phase_heatmap_inv = 1.0 / (phase_heatmap_norm + 1e-8)

            num_valid = torch.numel(pred)
            # RMSE / MAE / dB

            diff_norm = phase_pred_norm - phase_heatmap_norm
            diff_abs_norm = torch.abs(diff_norm)

            diff = pred - heatmap
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)
            # 100?
            rad_mean = torch.nanmean(diff_abs_norm)
            rad_median = torch.nanmedian(diff_abs_norm)

            # iRMSE / iMAE
            diff_inv = phase_pred_inv - phase_heatmap_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (heatmap + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = heatmap / (pred + 1e-8)
            r2 = pred / (heatmap + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25 ** 2).type_as(ratio)
            del_3 = (ratio < 1.25 ** 3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [rad_mean, rad_median, len_mean, len_median, rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()
        return result

    def evaluate_joint(self, sample, output, mode):
        with torch.no_grad():
            amp_pred = output['amp_pred'].detach()
            phase_pred = output['phase_pred'].detach()

            freq = sample['freq_level'].detach()  # 77
            freq *= 10 ** 9
            wave_len = 3e8 / freq
            wave_len = wave_len.view(freq.size(0), 1, 1, 1)

            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')

            amp_heatmap = output['amp_heatmap'].detach()
            phase_heatmap = output['phase_heatmap'].detach()

            # heatmap = heatmap[0, 0, :, :].data.cpu().numpy()
            # heatmap = (255.0 * cm(heatmap)).astype('uint8')
            assert not torch.isnan(amp_pred).any().item(), amp_pred
            assert not torch.isnan(amp_heatmap).any().item(), amp_heatmap

            amp_pred_norm = torch.from_numpy(normalize_data(amp_pred.cpu(), target_range=(-57, -30))).to('cuda')
            amp_heatmap_norm = torch.from_numpy(normalize_data(amp_heatmap.cpu(), target_range=(-57, -30))).to('cuda')
            assert not torch.isnan(amp_pred_norm).any().item(), amp_pred_norm
            assert not torch.isnan(amp_heatmap_norm).any().item(), amp_heatmap_norm

            # Do not normalize phase
            # phase_pred_norm = torch.from_numpy(normalize_data(phase_pred.cpu(), target_range=(1, 30))).to('cuda')
            #  phase_heatmap_norm = torch.from_numpy(normalize_data(phase_heatmap.cpu(), target_range=(1, 30))).to('cuda')
            phase_pred_norm = phase_pred
            phase_heatmap_norm = phase_heatmap
            assert not torch.isnan(phase_pred_norm).any().item(), phase_pred_norm
            assert not torch.isnan(phase_heatmap_norm).any().item(), phase_heatmap_norm

            # pathlength into phase
            phase_pred_norm = 2 * math.pi * phase_pred_norm / wave_len
            phase_heatmap_norm = 2 * math.pi * phase_heatmap_norm / wave_len

            phase_mask = (phase_heatmap_norm != float('inf')) & (phase_heatmap_norm != -float('inf'))
            phase_pred_norm = phase_pred_norm[phase_mask]
            phase_heatmap_norm = phase_heatmap_norm[phase_mask]

            # unwrap into wrap (0-2pi)
            phase_pred_norm = phase_pred_norm % (2 * math.pi)
            phase_heatmap_norm = phase_heatmap_norm % (2 * math.pi)

            amp_pred_inv = 1.0 / (amp_pred_norm + 1e-8)
            amp_heatmap_inv = 1.0 / (amp_heatmap_norm + 1e-8)

            phase_pred_inv = 1.0 / (phase_pred_norm + 1e-8)
            phase_heatmap_inv = 1.0 / (phase_heatmap_norm + 1e-8)
            # For numerical stability
            # mask = heatmap > self.t_valid
            # num_valid = mask.sum()

            # pred = pred[mask]
            # heatmap = heatmap[mask]

            # pred_inv = pred_inv[mask]
            # heatmap_inv = heatmap_inv[mask]

            # pred_inv[pred <= self.t_valid] = 0.0
            # heatmap_inv[heatmap <= self.t_valid] = 0.0
            amp_num_valid = torch.numel(amp_pred_norm)
            phase_num_valid = torch.numel(phase_pred_norm)

            # RMSE / MAE / dB
            amp_diff_norm = amp_pred_norm - amp_heatmap_norm
            amp_diff_abs_norm = torch.abs(amp_diff_norm)
            phase_diff_norm = phase_pred_norm - phase_heatmap_norm
            phase_diff_abs_norm = torch.abs(phase_diff_norm)

            amp_diff = amp_pred_norm - amp_heatmap_norm
            amp_diff_abs = torch.abs(amp_diff)
            amp_diff_sqr = torch.pow(amp_diff, 2)
            phase_diff = phase_pred_norm - phase_heatmap_norm
            phase_diff_abs = torch.abs(phase_diff)
            phase_diff_sqr = torch.pow(phase_diff, 2)

            amp_rmse = amp_diff_sqr.sum() / (amp_num_valid + 1e-8)
            amp_rmse = torch.sqrt(amp_rmse)
            amp_mae = amp_diff_abs.sum() / (amp_num_valid + 1e-8)
            phase_rmse = phase_diff_sqr.sum() / (phase_num_valid + 1e-8)
            phase_rmse = torch.sqrt(phase_rmse)
            phase_mae = phase_diff_abs.sum() / (phase_num_valid + 1e-8)

            # 100?
            amp_dB_mean = torch.nanmean(amp_diff_abs_norm)
            amp_dB_median = torch.nanmedian(amp_diff_abs_norm)
            phase_dB_mean = torch.nanmean(phase_diff_abs_norm)
            phase_dB_median = torch.nanmedian(phase_diff_abs_norm)

            # iRMSE / iMAE
            amp_diff_inv = amp_pred_inv - amp_heatmap_inv
            amp_diff_inv_abs = torch.abs(amp_diff_inv)
            amp_diff_inv_sqr = torch.pow(amp_diff_inv, 2)
            phase_diff_inv = phase_pred_inv - phase_heatmap_inv
            phase_diff_inv_abs = torch.abs(phase_diff_inv)
            phase_diff_inv_sqr = torch.pow(phase_diff_inv, 2)

            amp_irmse = amp_diff_inv_sqr.sum() / (amp_num_valid + 1e-8)
            amp_irmse = torch.sqrt(amp_irmse)
            phase_irmse = phase_diff_inv_sqr.sum() / (phase_num_valid + 1e-8)
            phase_irmse = torch.sqrt(phase_irmse)

            amp_imae = amp_diff_inv_abs.sum() / (amp_num_valid + 1e-8)
            phase_imae = phase_diff_inv_abs.sum() / (phase_num_valid + 1e-8)

            # Rel
            amp_rel = amp_diff_abs / (amp_heatmap_norm + 1e-8)
            amp_rel = amp_rel.sum() / (amp_num_valid + 1e-8)
            phase_rel = phase_diff_abs / (phase_heatmap_norm + 1e-8)
            phase_rel = phase_rel.sum() / (phase_num_valid + 1e-8)

            # delta
            amp_r1 = amp_heatmap_norm / (amp_pred_norm + 1e-8)
            amp_r2 = amp_pred_norm / (amp_heatmap_norm + 1e-8)
            amp_ratio = torch.max(amp_r1, amp_r2)
            phase_r1 = phase_heatmap_norm / (phase_pred_norm + 1e-8)
            phase_r2 = phase_pred_norm / (phase_heatmap_norm + 1e-8)
            phase_ratio = torch.max(phase_r1, phase_r2)

            amp_del_1 = (amp_ratio < 1.25).type_as(amp_ratio)
            amp_del_2 = (amp_ratio < 1.25 ** 2).type_as(amp_ratio)
            amp_del_3 = (amp_ratio < 1.25 ** 3).type_as(amp_ratio)
            phase_del_1 = (phase_ratio < 1.25).type_as(phase_ratio)
            phase_del_2 = (phase_ratio < 1.25 ** 2).type_as(phase_ratio)
            phase_del_3 = (phase_ratio < 1.25 ** 3).type_as(phase_ratio)

            amp_del_1 = amp_del_1.sum() / (amp_num_valid + 1e-8)
            amp_del_2 = amp_del_2.sum() / (amp_num_valid + 1e-8)
            amp_del_3 = amp_del_3.sum() / (amp_num_valid + 1e-8)
            phase_del_1 = phase_del_1.sum() / (phase_num_valid + 1e-8)
            phase_del_2 = phase_del_2.sum() / (phase_num_valid + 1e-8)
            phase_del_3 = phase_del_3.sum() / (phase_num_valid + 1e-8)

            result = [amp_dB_mean, amp_dB_median, amp_rmse, amp_mae, amp_irmse, amp_imae, amp_rel, amp_del_1, amp_del_2, amp_del_3]
            result.extend([phase_dB_mean, phase_dB_median, phase_rmse, phase_mae, phase_irmse, phase_imae, phase_rel, phase_del_1, phase_del_2, phase_del_3])

            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()
        return result

    # lili video
    def evaluate_real_video(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("before: ", hist_percentage)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            errors = []
            mmwave_errors = []
            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            for fps in range(10):
                result = np.load(dir_real + real_name + "/mmWave_sim_video/{x}/".format(x=fps) + "results.npy")
                coord = np.load(dir_real + real_name + "/mmWave_sim_video/{x}/".format(x=fps) + "/coords.npy")
                AP = np.load(dir_real + real_name + "/mmWave_sim_video/{x}/".format(x=fps) + "/AP.npy")

                json_name = 'rss_{x}.json'.format(x=fps)
                with open(dir_real + real_name + "/" + json_name, 'r') as file:
                    # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                    data = json.load(file)

                x_real = [entry["x"] for entry in data]
                y_real = [entry["y"] for entry in data]
                med_rssi_real = [entry["med_rssi"] for entry in data]
                med_rssi_real = np.clip(med_rssi_real, -70, -30)

                # print(np.min(med_rssi_real), np.max(med_rssi_real))
                hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                               -55, -50,
                                                               -45, -40,
                                                               -35, -30, -25])
                hist_percentage = (hist / np.sum(hist)) * 100
                print(hist_percentage)
                # print(hist_percentage, bins)
                # print(med_rssi_real.shape)

                # pred = averaged_tensor.squeeze().numpy()
                best_error = 100000
                # pred = np.clip(pred_origin, 0.005, 0.95)
                # from skimage import exposure
                pred = pred_origin
                # pred = exposure.match_histograms(pred, result)
                pred = normalize_data(pred, target_range=(np.min(med_rssi_real), np.max(med_rssi_real)))

                # pred = np.round(pred)
                # print(np.min(pred), np.max(pred))
                hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                      -55, -50,
                                                      -45, -40,
                                                      -35, -30, -25])
                hist_percentage = (hist / np.sum(hist)) * 100
                print("after: ", hist_percentage)
                # print(hist_percentage, bins)

                # print(pred.shape)
                x_real_norm = normalize_data(x_real, (0, pred.shape[0] - 1))
                y_real_norm = normalize_data(y_real, (0, pred.shape[1] - 1))

                # result = np.clip(result, a_min=np.min(med_rssi_real), a_max=np.max(med_rssi_real))
                result = normalize_data(result, target_range=(np.min(med_rssi_real), np.max(med_rssi_real)))

                assert len(x_real) == len(y_real) == len(med_rssi_real)

                for idx in range(len(x_real)):
                    value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                    predict_rssi = result[value[0]][value[1]]
                    # med_rssi_real[idx]
                    # pre[x_real_norm[idx]][y_real_norm[idx]]

                    mmwave_error = abs(med_rssi_real[idx] - predict_rssi)
                    error = abs(med_rssi_real[idx] - pred[round(x_real_norm[idx])][round(y_real_norm[idx])])

                    # print(error, final_error, med_rssi_real[idx], coord_indices, (x_real[idx], y_real[idx]))
                    errors.append(error)
                    mmwave_errors.append(mmwave_error)
            np.save("/home/yifanyang/mmwave_result.npy", mmwave_errors)
            np.save("/home/yifanyang/diffusion_result.npy", errors)

            print("Diffusion error: ", np.nanmean(errors), np.nanmedian(errors))
            print("total: ", len(errors))
            errors_5 = remove_outliers(errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in errors if x < 1) / len(errors) * 100)
            print("==")
            print("MMwave error: ", np.nanmean(mmwave_errors), np.nanmedian(mmwave_errors))
            errors_5 = remove_outliers(mmwave_errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(mmwave_errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in mmwave_errors if x < 1) / len(mmwave_errors) * 100)

            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

    # lili image
    def evaluate_real_lili_image(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("before: ", hist_percentage)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            result = np.load(dir_real + real_name + "/results.npy")

            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            coord = np.load(dir_real + real_name + "/coords.npy")
            AP = np.load(dir_real + real_name + "/AP.npy")

            # pred_origin = reshape_with_average_pooling(pred_origin, result.shape[0], result.shape[1])
            if real_name == 'apartment':
                json_name = 'exp_apartment-rssi-0817-codebook.json'
            elif real_name == '4321':
                json_name = 'exp_rssi-4321-retest-0824.json'
                # json_name = '4321_exp_rssi.json'
            elif real_name == 'lili':
                json_name = 'lili-0420.json'
            else:
                # raise Exception(real_name)
                result = [torch.tensor(0)] * 11
                result = torch.stack(result)
                result = torch.unsqueeze(result, dim=0).detach()
                return result

            with open(dir_real + "data/" + json_name, 'rb') as file:
                # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                data = pickle.load(file)

            x_real = [entry["x"] for entry in data]
            y_real = [entry["y"] for entry in data]
            med_rssi_real = [entry["rss"] for entry in data]
            med_rssi_real = np.clip(med_rssi_real, -70, -30)

            # print(np.min(med_rssi_real), np.max(med_rssi_real))
            hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                           -55, -50,
                                                           -45, -40,
                                                           -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print(hist_percentage)
            # print(hist_percentage, bins)
            # print(med_rssi_real.shape)

            # pred = averaged_tensor.squeeze().numpy()
            best_error = 100000
            # pred = np.clip(pred_origin, 0.005, 0.95)
            # from skimage import exposure
            pred = pred_origin
            # pred = exposure.match_histograms(pred, result)
            # pred = np.clip(pred, a_min=-57, a_max=-30)
            pred = normalize_data(pred, target_range=(np.min(result), np.max(result)))
            pred = normalize_data(pred, target_range=(np.min(med_rssi_real), np.max(med_rssi_real)))

            # pred = np.round(pred)
            # print(np.min(pred), np.max(pred))
            hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                  -55, -50,
                                                  -45, -40,
                                                  -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("after: ", hist_percentage)
            # print(hist_percentage, bins)

            # print(pred.shape)
            x_real_norm = normalize_data(x_real, (0, pred.shape[0] - 1))
            y_real_norm = normalize_data(y_real, (0, pred.shape[1] - 1))

            # result = np.clip(result, a_min=-57, a_max=-30)
            #  result = normalize_data(result, target_range=(-57, -30))

            result = normalize_data(result, target_range=(np.min(med_rssi_real), np.max(med_rssi_real)))

            assert len(x_real) == len(y_real) == len(med_rssi_real)
            errors = []

            mmwave_errors = []

            for idx in range(len(x_real)):
                value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                predict_rssi = result[value[0]][value[1]]
                # med_rssi_real[idx]
                # pre[x_real_norm[idx]][y_real_norm[idx]]

                mmwave_error = abs(med_rssi_real[idx] - predict_rssi)
                """
                best_tmp_error = 100
                for x_tmp in range(-5, 4):
                    for y_tmp in range(-5, 4):
                        predict_rssi = pred[min(max(round(x_real_norm[idx]) + x_tmp, 0), 351)][min(max(round(y_real_norm[idx]) + y_tmp, 0), 705)]

                        error = abs(predict_rssi - med_rssi_real[idx])
                        if error < best_tmp_error:
                            final_error = error
                            best_tmp_error = final_error
                """
                error = pred[min(max(round(x_real_norm[idx]), 0), 351)][min(max(round(y_real_norm[idx]), 0), 705)]

                # predict_rssi /= 25
                # true_rssi = med_rssi_real[idx]
                # error = abs(predict_rssi - true_rssi)
                error = final_error

                # print(error, final_error, med_rssi_real[idx], coord_indices, (x_real[idx], y_real[idx]))
                errors.append(error)
                mmwave_errors.append(mmwave_error)
            print(errors)
            print("diffusion error: ", np.nanmean(errors), np.nanmedian(errors))
            errors_5 = remove_outliers(errors, 10)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(errors, 20)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in errors if x < 1) / len(errors) * 100)

            print("==")
            print(mmwave_errors)
            print("MMwave error: ", np.nanmean(mmwave_errors), np.nanmedian(mmwave_errors))
            errors_5 = remove_outliers(mmwave_errors, 10)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(mmwave_errors, 20)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in mmwave_errors if x < 1) / len(mmwave_errors) * 100)

            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

    def evaluate_real_tmp3(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("before: ", hist_percentage)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            result = np.load(dir_real + real_name + "/results.npy")

            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            coord = np.load(dir_real + real_name + "/coords.npy")
            AP = np.load(dir_real + real_name + "/AP.npy")

            # pred_origin = reshape_with_average_pooling(pred_origin, result.shape[0], result.shape[1])
            if real_name == 'apartment':
                json_name = 'exp_apartment-rssi-0817-codebook.json'
            elif real_name == '4321':
                json_name = 'exp_rssi-4321-retest-0824.json'
                # json_name = '4321_exp_rssi.json'
            elif real_name == 'lili':
                json_name = 'lili-0420.json'
            else:
                raise Exception(real_name)

            with open(dir_real + "data/" + json_name, 'r') as file:
                # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                data = json.load(file)

            x_real = [entry["x"] for entry in data]
            y_real = [entry["y"] for entry in data]
            med_rssi_real = [entry["med_rssi"] for entry in data]
            med_rssi_real = np.clip(med_rssi_real, -70, -30)

            # print(np.min(med_rssi_real), np.max(med_rssi_real))
            hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                           -55, -50,
                                                           -45, -40,
                                                           -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print(hist_percentage)
            # print(hist_percentage, bins)
            # print(med_rssi_real.shape)

            # pred = averaged_tensor.squeeze().numpy()
            best_error = 100000
            # pred = np.clip(pred_origin, 0.005, 0.95)
            # from skimage import exposure
            pred = pred_origin
            # pred = exposure.match_histograms(pred, result)
            pred = normalize_data(pred, target_range=(-57, -30))

            pred = np.round(pred)
            # print(np.min(pred), np.max(pred))
            hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                  -55, -50,
                                                  -45, -40,
                                                  -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("after: ", hist_percentage)
            # print(hist_percentage, bins)

            # print(pred.shape)
            x_real_norm = normalize_data(x_real, (0, pred.shape[0] - 1))
            y_real_norm = normalize_data(y_real, (0, pred.shape[1] - 1))

            result = np.clip(result, a_min=-57, a_max=-30)

            assert len(x_real) == len(y_real) == len(med_rssi_real)
            errors = []

            mmwave_errors = []

            for idx in range(len(x_real)):
                # value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                # predict_rssi = result[value[0]][value[1]]
                mmwave_coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord, threshold=0.04)
                diffusion_coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord, threshold=0.05)

                predict_rssi = 0
                mmwave_rssi = 0
                mmwave_count = 0
                diffusion_count = 0
                best_tmp_error = 100
                final_rssi = 0
                print(coord.shape)
                print(x_real[idx], y_real[idx], med_rssi_real[idx])
                for x_ind, y_ind in zip(mmwave_coord_indices[0], mmwave_coord_indices[1]):
                    print(coord[x_ind][y_ind], result[x_ind][y_ind])

                    # mmwave_count += 1
                    # mmwave_rssi += result[x_ind][y_ind]

                    error = abs(result[x_ind][y_ind] - med_rssi_real[idx])

                    if error < best_tmp_error:
                        final_rssi = result[x_ind][y_ind]
                        best_tmp_error = error
                mmwave_rssi = final_rssi
                print("==")
                best_tmp_error = 100
                for x_tmp in range(-5, 4):
                    for y_tmp in range(-5, 4):
                        predict_rssi = pred[min(max(round(x_real_norm[idx]) + x_tmp, 0), 351)][min(max(round(y_real_norm[idx]) + y_tmp, 0), 705)]

                        error = abs(predict_rssi - med_rssi_real[idx])
                        if error < best_tmp_error:
                            final_error = error
                            best_tmp_error = final_error
                # predict_rssi /= 25
                # true_rssi = med_rssi_real[idx]
                # error = abs(predict_rssi - true_rssi)
                error = final_error
                # mmwave_rssi /= mmwave_count
                if med_rssi_real[idx] <= -57:
                    mmwave_error = 0.0
                else:
                    mmwave_error = abs(mmwave_rssi - med_rssi_real[idx])
                # print(error, final_error, med_rssi_real[idx], coord_indices, (x_real[idx], y_real[idx]))
                errors.append(error)
                mmwave_errors.append(mmwave_error)

            print("total: ", len(errors))
            errors_5 = remove_outliers(errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in errors if x < 1) / len(errors) * 100)

            print("==")
            print("MMwave error: ", np.nanmean(mmwave_errors), np.nanmedian(mmwave_errors))
            errors_5 = remove_outliers(mmwave_errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(mmwave_errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in mmwave_errors if x < 1) / len(mmwave_errors) * 100)

            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

    # apartment
    def evaluate_real(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("before: ", hist_percentage)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            result = np.load(dir_real + real_name + "/results.npy")

            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            coord = np.load(dir_real + real_name + "/coords.npy")
            AP = np.load(dir_real + real_name + "/AP.npy")

            # pred_origin = reshape_with_average_pooling(pred_origin, result.shape[0], result.shape[1])
            if real_name == 'apartment':
                json_name = 'exp_apartment-rssi-0817-codebook.json'
            elif real_name == '4321':
                json_name = 'exp_rssi-4321-retest-0824.json'
            elif real_name == 'lili':
                json_name = 'lili-0420.json'
            else:
                raise Exception

            with open(dir_real + "data/" + json_name, 'r') as file:
                # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                data = json.load(file)

            x_real = [entry["x"] for entry in data]
            y_real = [entry["y"] for entry in data]
            med_rssi_real = [entry["med_rssi"] for entry in data]
            med_rssi_real = np.clip(med_rssi_real, -70, -30)

            # print(np.min(med_rssi_real), np.max(med_rssi_real))
            hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                           -55, -50,
                                                           -45, -40,
                                                           -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print(hist_percentage)
            # print(hist_percentage, bins)
            # print(med_rssi_real.shape)

            # pred = averaged_tensor.squeeze().numpy()
            best_error = 100000
            # pred = np.clip(pred_origin, 0.005, 0.95)
            # from skimage import exposure
            pred = pred_origin

            # pred = exposure.match_histograms(pred, result)
            pred = normalize_data(pred, target_range=(-57, -30))

            pred = np.round(pred)
            # print(np.min(pred), np.max(pred))
            hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                  -55, -50,
                                                  -45, -40,
                                                  -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100
            print("after: ", hist_percentage)
            # print(hist_percentage, bins)

            # print(pred.shape)
            x_real_norm = normalize_data(x_real, (0, pred.shape[0] - 1))
            y_real_norm = normalize_data(y_real, (0, pred.shape[1] - 1))

            result = np.clip(result, a_min=-70, a_max=-30)

            assert len(x_real) == len(y_real) == len(med_rssi_real)
            errors = []

            mmwave_errors = []

            for idx in range(len(x_real)):
                # value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                # predict_rssi = result[value[0]][value[1]]
                mmwave_coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord, threshold=0.1)
                diffusion_coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord, threshold=0.05)

                predict_rssi = 0
                mmwave_rssi = 0
                mmwave_count = 0
                diffusion_count = 0
                best_tmp_error = 100
                final_rssi = 0
                print(coord.shape)
                print(x_real[idx], y_real[idx], med_rssi_real[idx])
                for x_ind, y_ind in zip(mmwave_coord_indices[0], mmwave_coord_indices[1]):
                    print(coord[x_ind][y_ind], result[x_ind][y_ind])

                    # mmwave_count += 1
                    # mmwave_rssi += result[x_ind][y_ind]

                    error = abs(result[x_ind][y_ind] - med_rssi_real[idx])

                    if error < best_tmp_error:
                        final_rssi = result[x_ind][y_ind]
                        best_tmp_error = error
                mmwave_rssi = final_rssi
                print("==")
                best_tmp_error = 100
                for x_tmp in range(-5, 4):
                    for y_tmp in range(-5, 4):
                        predict_rssi = pred[min(max(round(x_real_norm[idx]) + x_tmp, 0), 351)][min(max(round(y_real_norm[idx]) + y_tmp, 0), 705)]

                        error = abs(predict_rssi - med_rssi_real[idx])
                        if error < best_tmp_error:
                            final_error = error
                            best_tmp_error = final_error
                # predict_rssi /= 25
                # true_rssi = med_rssi_real[idx]
                # error = abs(predict_rssi - true_rssi)
                error = final_error
                # mmwave_rssi /= mmwave_count
                if med_rssi_real[idx] <= -57:
                    mmwave_error = 0.0
                else:
                    mmwave_error = abs(mmwave_rssi - med_rssi_real[idx])
                # print(error, final_error, med_rssi_real[idx], coord_indices, (x_real[idx], y_real[idx]))
                errors.append(error)
                mmwave_errors.append(mmwave_error)

            print("total: ", len(errors))
            errors_5 = remove_outliers(errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in errors if x < 1) / len(errors) * 100)

            print("==")
            print("MMwave error: ", np.nanmean(mmwave_errors), np.nanmedian(mmwave_errors))
            errors_5 = remove_outliers(mmwave_errors, 5)
            print("errors_5: ", np.nanmean(errors_5), np.nanmedian(errors_5))
            errors_10 = remove_outliers(mmwave_errors, 10)
            print("errors_10: ", np.nanmean(errors_10), np.nanmedian(errors_10))
            print(sum(1 for x in mmwave_errors if x < 1) / len(mmwave_errors) * 100)

            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

    def evaluate_real_tmp(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            # print(hist, bins)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            result = np.load(dir_real + real_name + "/results.npy")

            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            coord = np.load(dir_real + real_name + "/coords.npy")
            # print(coord.shape)
            # print(coord[0][0], coord[0][1], coord[0][2])
            # print(coord[1][0], coord[2][0], coord[3][0])
            AP = np.load(dir_real + real_name + "/AP.npy")
            # heatmap = heatmap[0, 0, :, :].data.cpu().numpy()
            # heatmap = (255.0 * cm(heatmap)).astype('uint8')
            # Calculate the kernel sizes for average pooling

            # pred_origin = reshape_with_average_pooling(pred_origin, result.shape[0], result.shape[1])
            if real_name == 'apartment':
                json_name = 'exp_apartment-rssi-0817-codebook.json'
            elif real_name == '4321':
                json_name = 'exp_rssi-4321-retest-0824.json'
            else:
                raise Exception

            with open(dir_real + "data/" + json_name, 'r') as file:
                # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                data = json.load(file)

            x_real = [entry["x"] for entry in data]
            y_real = [entry["y"] for entry in data]
            med_rssi_real = [entry["med_rssi"] for entry in data]
            med_rssi_real = np.clip(med_rssi_real, -57, -30)
            # print(np.min(med_rssi_real), np.max(med_rssi_real))
            hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                           -55, -50,
                                                           -45, -40,
                                                           -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100

            # print(hist_percentage, bins)
            # print(med_rssi_real.shape)

            # pred = averaged_tensor.squeeze().numpy()
            best_error = 100000
            print("pred_origin", pred_origin)
            for min_ratio in range(51):
                for max_ratio in range(11):
                    errors = []
                    min_value = 0.0 + 0.5 * min_ratio / 50
                    max_value = 0.8 + 0.2 * max_ratio / 10
                    pred = np.clip(pred_origin, min_value, max_value)
                    # from skimage import exposure

                    # pred = exposure.match_histograms(pred, result)
                    pred = normalize_data(pred, target_range=(-57, -30))

                    pred = np.round(pred)
                    # print(np.min(pred), np.max(pred))

                    hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                          -55, -50,
                                                          -45, -40,
                                                          -35, -30, -25])
                    hist_percentage = (hist / np.sum(hist)) * 100

                    # print(hist_percentage, bins)

                    # print(pred.shape)
                    x_real_norm = normalize_data(x_real, (0, pred.shape[0] - 1))
                    y_real_norm = normalize_data(y_real, (0, pred.shape[1] - 1))

                    assert len(x_real) == len(y_real) == len(med_rssi_real)
                    for idx in range(len(x_real)):
                        # value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                        # predict_rssi = result[value[0]][value[1]]
                        coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord)

                        predict_rssi = 0
                        if not coord_indices:
                            print("coord_indices doesn't exist")
                            continue
                        # count = 0
                        # print(pred.shape)
                        # print(np.min(x_real_norm), np.max(x_real_norm))
                        best_tmp_error = 100
                        agg_error = 0
                        for x_tmp in range(-1, 2):
                            for y_tmp in range(-1, 2):
                                predict_rssi = pred[min(max(round(x_real_norm[idx]) + x_tmp, 0), 351)][min(max(round(y_real_norm[idx]) + y_tmp, 0), 705)]
                                error = abs(predict_rssi - med_rssi_real[idx])
                                agg_error += error
                        agg_error /= 9
                        # predict_rssi /= 25
                        # true_rssi = med_rssi_real[idx]
                        # error = abs(predict_rssi - true_rssi)
                        # print(error, final_error, med_rssi_real[idx], coord_indices, (x_real[idx], y_real[idx]))
                        # errors.append(error)
                        errors.append(agg_error)

                    # print(errors)
                    # print(np.nanmean(errors))
                    # print(np.nanmedian(errors))
                    # errors = remove_outliers(errors, 10)
                    if np.nanmean(errors) < best_error:
                        print("found the better")
                        print(min_value, max_value)
                        print(np.nanmean(errors))
                        print(np.nanmedian(errors))
                        best_error = np.nanmean(errors)

            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

    def evaluate_real_tmp2(self, dir_real, real_name, output):
        with torch.no_grad():
            # (352, 706)
            # pred_origin = output['pred'][0, 0, 40:320, 40:600].detach().cpu().numpy()
            pred_origin = output['pred'][0, 0, :, :].detach().cpu().numpy()
            # print(np.min(pred_origin), np.max(pred_origin))
            # print(pred_origin.shape)
            hist, bins = np.histogram(pred_origin, bins=[0, 0.1,
                                                         0.2, 0.3,
                                                         0.4, 0.5,
                                                         0.6, 0.7,
                                                         0.8, 0.9,
                                                         1])
            # print(hist, bins)
            # pred = pred[0, 0, :, :].data.cpu().numpy()
            # pred = (255.0 * cm(pred)).astype('uint8')
            result = np.load(dir_real + real_name + "/results.npy")
            x_ratio = pred_origin.shape[0] / result.shape[0]
            y_ratio = pred_origin.shape[1] / result.shape[1]
            # print(x_ratio)
            # print(y_ratio)
            # print(result.shape)
            coord = np.load(dir_real + real_name + "/coords.npy")
            # print(coord.shape)
            # print(coord[0][0], coord[0][1], coord[0][2])
             #print(coord[1][0], coord[2][0], coord[3][0])
            AP = np.load(dir_real + real_name + "/AP.npy")
            # heatmap = heatmap[0, 0, :, :].data.cpu().numpy()
            # heatmap = (255.0 * cm(heatmap)).astype('uint8')
            # Calculate the kernel sizes for average pooling
            kernel_size = (
                pred_origin.shape[0] // result.shape[0],
                pred_origin.shape[1] // result.shape[1]
            )
            # input_tensor = torch.from_numpy(pred_origin).unsqueeze(0).unsqueeze(0)
            # Perform average pooling
            # averaged_tensor = torch.nn.functional.avg_pool2d(input_tensor, kernel_size=kernel_size)

            pred_origin = reshape_with_average_pooling(pred_origin, result.shape[0], result.shape[1])
            if real_name == 'apartment':
                json_name = 'exp_apartment-rssi-0817-codebook.json'
            elif real_name == '4321':
                json_name = 'exp_rssi-4321-retest-0824.json'
            else:
                raise Exception

            with open(dir_real + "data/" + json_name, 'r') as file:
                # with open(input_path + "data/exp_rssi-4321-retest-0824.json", 'r') as file:
                data = json.load(file)

            x_real = [entry["x"] for entry in data]
            y_real = [entry["y"] for entry in data]
            med_rssi_real = [entry["med_rssi"] for entry in data]
            med_rssi_real = np.clip(med_rssi_real, -57, -30)
            # print(np.min(med_rssi_real), np.max(med_rssi_real))
            hist, bins = np.histogram(med_rssi_real, bins=[-60, -56.999,
                                                           -55, -50,
                                                           -45, -40,
                                                           -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100

            # print(hist_percentage, bins)
            # print(med_rssi_real.shape)

            # pred = averaged_tensor.squeeze().numpy()
            best_error = 100000

            errors = []
            min_value = 0.7
            max_value = 0.9
            pred = np.clip(pred_origin, min_value, max_value)
            # from skimage import exposure

            # pred = exposure.match_histograms(pred, result)

            pred = normalize_data(pred, target_range=(-57, -30))

            pred = np.round(pred)
            # print(np.min(pred), np.max(pred))

            hist, bins = np.histogram(pred, bins=[-60, -56.999,
                                                  -55, -50,
                                                  -45, -40,
                                                  -35, -30, -25])
            hist_percentage = (hist / np.sum(hist)) * 100

            # print(hist_percentage, bins)

            # print(pred.shape)

            assert len(x_real) == len(y_real)
            for idx in range(len(x_real)):
                # value, index = find_closest_coordinate(x_real[idx], y_real[idx], coord)
                # predict_rssi = result[value[0]][value[1]]
                coord_indices = find_matching_coordinates(x_real[idx], y_real[idx], coord)

                predict_rssi = 0
                if not coord_indices:
                    print("coord_indices doesn't exist")
                    continue
                count = 0
                for x_ind, y_ind in zip(coord_indices[0], coord_indices[1]):
                    if pred[x_ind][y_ind] is not np.nan:
                        # adj_x_ind = int(x_ind * x_ratio)
                        # adj_y_ind = int(y_ind * y_ratio)
                        # predict_rssi += pred[adj_x_ind][adj_y_ind]
                        predict_rssi += pred[x_ind][y_ind]
                        count += 1
                    else:
                        continue
                predict_rssi /= count
                true_rssi = med_rssi_real[idx]
                error = abs(predict_rssi - true_rssi)
                # print(error, predict_rssi, true_rssi, coord_indices, (x_real[idx], y_real[idx]))
                errors.append(error)

            # print(errors)
            # print(np.nanmean(errors))
            # print(np.nanmedian(errors))
            if np.nanmean(errors) < best_error:
                print("found the better")
                print(min_value, max_value)
                best_min_value = min_value
                best_max_value = max_value
                print(np.nanmean(errors))
                best_error = np.nanmean(errors)
            # print("finished")
            # print(best_error)
            # print(best_min_value, best_max_value)
            result = [torch.tensor(np.nanmean(errors)), torch.tensor(np.nanmedian(errors)), torch.tensor(0),
                      torch.tensor(0),
                      torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0),
                      torch.tensor(0)]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result
