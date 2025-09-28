# -*- coding: utf-8 -*-
"""NeRF2 runner for training and testing
"""

import os

from baseline_utils.data_painter import paint_spectrum_compare
from baseline_utils.logger import logger_config
from model.baseline.nerf2 import *
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from model.utils import normalize_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from shutil import copyfile
from PIL import Image

import numpy as np
import torch
import torch.optim as optim
import yaml
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model.renderer import renderer_dict
import json
import shutil


class NeRF2_Runner():
    def __init__(self, args) -> None:
        super(NeRF2_Runner, self).__init__()
        self.args = args
        self.args.config = 'config/ble-rssi.yml'
        self.args.dataset_type = 'ble'
        self.args.mode = 'train'
        # torch.cuda.set_device(args.gpu)

        ## backup config file
        if self.args.mode == 'train':
            logdir = args.save_dir
            os.makedirs(logdir, exist_ok=True)
            copyfile(args.config, os.path.join(logdir, 'config.yml'))

        self.args = args
        with open(self.args.split_json) as json_file:
            self.json_data = json.load(json_file)
        with open(self.args.config) as f:
            kwargs = yaml.safe_load(f)
            f.close()
        dataset_type = 'ble'
        kwargs_path = kwargs['path']
        self.kwargs_render = kwargs['render']
        self.kwargs_network = kwargs['networks']
        self.kwargs_train = kwargs['train']
        self.dataset_type = dataset_type
        self.eta = []

        ## Path settings
        self.expname = kwargs_path['expname']
        # self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.args.save_dir, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nerf2')
        self.logger.info("self.args.save_dir:%s", self.args.save_dir)
        self.writer = SummaryWriter(os.path.join(self.args.save_dir, 'tensorboard'))

        ## Networks
        self.init_train()

        with open(self.args.split_json) as json_file:
            self.json_data = json.load(json_file)
            self.sample_list = self.json_data['test']

        # train_set, test_set = self.split_dataset(ratio=0.8)
        self.logger.info("Loading training/test set...")
        error_list = []
        mean_ssim_list = []
        error_elems = []
        # data_dir = os.listdir(self.args.dir_data + "/output_heatmap_aug")[0]
        for idx in range(len(self.sample_list)):
            train_set = BLE_dataset(self.args.dir_data, self.sample_list[idx], self.scale_worldsize, total_iterations=self.total_iterations, fix=False)
            test_set = BLE_dataset(self.args.dir_data, self.sample_list[idx], self.scale_worldsize, total_iterations=self.total_iterations, test=True, fix=False)

            self.train_iter = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
            self.test_iter = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
            # self.logger.info("train size:%d, test size:%d", len(train_set), len(test_set))
            self.logger.info("test size:%d", len(test_set))
            # self.load_dataset()

            # self.operate_model()
            # print("Start train")
            self.init_train()
            self.train()
            # print("Start eval")

            mean_error, median_error, mean_ssim, _ = self.eval_network_rssi(idx, test_set.results_shape)
            error_list.append(mean_error)
            mean_ssim_list.append(mean_ssim)
            self.logger.info("Stacked Mean error:%.2f", np.mean(error_list))
            self.logger.info("Stacked Median error:%.2f", np.median(error_list))
            self.logger.info("Stacked SSIM:%.2f", np.mean(mean_ssim_list))
            np.save(os.path.join(self.args.save_dir, "error_elems.npy"), np.array(error_list))
            np.save(os.path.join(self.args.save_dir, "error_ssim_elems.npy"), np.array(mean_ssim_list))

        self.logger.info("Final Mean error:%.2f", np.mean(error_list))
        self.logger.info("Final Median error:%.2f", np.median(error_list))

    def init_train(self):
        self.nerf2_network = NeRF2(**self.kwargs_network).to(self.devices)
        params = list(self.nerf2_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(self.kwargs_train['lr']),
                                          weight_decay=float(self.kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                     T_max=float(self.kwargs_train['T_max']),
                                                                     eta_min=float(self.kwargs_train['eta_min']),
                                                                     last_epoch=-1)

        ## Renderer
        renderer = renderer_dict[self.kwargs_render['mode']]
        self.renderer = renderer(networks_fn=self.nerf2_network, **self.kwargs_render)
        self.scale_worldsize = self.kwargs_render['scale_worldsize']

        ## Print total number of parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        # if kwargs_train['load_ckpt'] or args.mode == 'test':
        #     self.load_checkpoints()
        self.batch_size = self.kwargs_train['batch_size']
        self.total_iterations = self.kwargs_train['total_iterations']
        self.save_freq = self.kwargs_train['save_freq']

    def split_dataset(self, ratio=0.8):
        """random shuffle train/test set
        """
        data_dir = os.listdir(self.args.dir_data + "/output_heatmap_aug")
        return data_dir[:int(len(data_dir) * 0.8)], data_dir[int(len(data_dir) * 0.8):]

    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.nerf2_network.load_state_dict(ckpt['nerf2_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=20,
                                                                         eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']

    def operate_model(self):
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'test':
            self.eval_network_rssi()

    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'nerf2_network_state_dict': self.nerf2_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname

    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current Iteration:%d", self.current_iteration)
        while self.current_iteration <= self.total_iterations:
            with tqdm(total=len(self.train_iter),
                      desc=f"Iteration {self.current_iteration}/{self.total_iterations}") as pbar:
                for train_input, train_label in self.train_iter:
                    if self.current_iteration > self.total_iterations:
                        break

                    train_input, train_label = train_input.to(self.devices), train_label.to(self.devices)
                    assert self.dataset_type == 'ble'

                    tx_o, rays_o, rays_d = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:]
                    predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d)
                    loss = img2mse(predict_rssi, train_label.view(-1))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.cosine_scheduler.step()
                    self.current_iteration += 1

                    self.writer.add_scalar('Loss/loss', loss, self.current_iteration)
                    pbar.update(1)
                    pbar.set_description(f"Iteration {self.current_iteration}/{self.total_iterations}")
                    pbar.set_postfix_str(
                        'loss = {:.6f}, lr = {:.6f}'.format(loss.item(), self.optimizer.param_groups[0]['lr']))

    def eval_network_rssi(self, idx, results_shape):
        """test the model and save predicted RSSI values to a file
        """
        # self.logger.info("Start evaluation")
        self.nerf2_network.eval()
        final_predict_rssi = []
        final_gt_rssi = []
        mean_error_list = []
        median_error_list = []
        ssim_error_list = []
        error_list = []
        sample_name = self.sample_list[idx]['heatmap']
        import time
        s_time = time.time()
        with torch.no_grad():
            with open(os.path.join(self.args.save_dir, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    predict_rssi_elem = []
                    gt_rssi_elem = []
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    tx_o, rays_o, rays_d = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:]

                    # Predict RSSI
                    predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    # print(torch.min(predict_rssi), torch.max(predict_rssi))
                    # predict_rssi = np.clip(predict_rssi, -57, -30)
                    predict_rssi_norm = torch.clamp(predict_rssi, min=-57, max=-30)
                    # print(predict_rssi.numpy())
                    # predict_rssi = torch.from_numpy(normalize_data(predict_rssi.numpy(), target_range=(-57, -30)))
                    # print(torch.min(predict_rssi), torch.max(predict_rssi))

                    gt_rssi = amplitude2rssi(test_label.detach().cpu())
                    gt_rssi = gt_rssi.reshape(-1)
                    gt_rssi_norm = torch.clamp(gt_rssi, min=-57, max=-30)
                    # print(gt_rssi.numpy())
                    # gt_rssi = torch.from_numpy(normalize_data(gt_rssi.numpy(), target_range=(-57, -30)))
                    # print(torch.min(gt_rssi), torch.max(gt_rssi))
                    # print("==")
                    # gt_rssi = np.clip(gt_rssi, -57, -30)

                    final_predict_rssi.extend(predict_rssi)
                    final_gt_rssi.extend(gt_rssi)
                    error = abs(predict_rssi_norm - gt_rssi_norm)
                    error_list.append(error)
                    # predict_rssi_elem.append(predict_rssi_norm)
                    # gt_rssi_elem.append(gt_rssi_norm)
                    mean_error_list.append(torch.mean(error))
                    median_error_list.append(torch.median(error))
                    predict_rssi_norm = np.abs(predict_rssi_norm.numpy())
                    gt_rssi_norm = np.abs(gt_rssi_norm.numpy())
                    try:
                        ssim_value = ssim(predict_rssi_norm, gt_rssi_norm, data_range=70)
                    except ValueError:
                        ssim_value = 1
                    ssim_error_list.append(ssim_value)

                    # self.logger.info("Mean error:%.2f", mean_error)
                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')
        self.eta.append(time.time() - s_time)
        if len(self.eta) == 100:
            np.save("nerf_runtime.npy", self.eta)
        # print(len(final_predict_rssi[0]))
        # final_predict_rssi = np.resize(final_predict_rssi, (136, 145))
        # final_gt_rssi = np.resize(final_gt_rssi, (136, 145))
        final_predict_rssi = np.reshape(final_predict_rssi, results_shape)
        final_gt_rssi = np.reshape(final_gt_rssi, results_shape)

        os.makedirs(os.path.join(self.args.save_dir, str(idx)))
        self.save_heatmap(final_predict_rssi, os.path.join(self.args.save_dir, str(idx), "prediced_rssi.png"))
        self.save_heatmap(final_gt_rssi, os.path.join(self.args.save_dir, str(idx), "gt_rssi.png"))


        # Copy the file
        shutil.copy(sample_name, self.args.save_dir)

        with open(os.path.join(self.args.save_dir, str(idx), "room_name.txt"), 'w') as f:
            f.write(sample_name)
            f.write(str(np.mean(mean_error_list)))
            f.write(str(np.mean(median_error_list)))

        self.logger.info("Current Mean error:%.2f", np.mean(mean_error_list))
        self.logger.info("Current Median error:%.2f", np.mean(median_error_list))

        return np.mean(mean_error_list), np.mean(median_error_list), np.mean(ssim_error_list), error_list

    def save_heatmap(self, result, file_name):
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


def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)


class BLE_dataset(Dataset):
    """ble dataset class
    """

    def __init__(self, dir_data, datadir, scale_worldsize=1, total_iterations=3000, test=False, fix=False) -> None:
        super().__init__()

        self.datadir = datadir
        self.results = []
        self.coords = []
        self.AP = []
        self.gateway_coords = []
        self.rssis = []
        patch_span = 3
        if fix:
            self.n_gateways = 15 * patch_span * patch_span # For pixel-specific measurement
        else:
            self.n_gateways = 5
        self.results_shape = None
        # self.n_gateways = 15 * 10  # For larger measurement
        # We don't need larger measuremed because it is already dimension-reduced version (252x172 not 1920x1080) due to usage of results.npy
        self.is_test = test
        coordinates = []

        heatmap_path = os.path.join(dir_data, datadir['heatmap'])
        sub_path = "/".join(heatmap_path.split("/")[:-1])
        if os.path.exists(sub_path + "/results.npy") and os.path.exists(
                sub_path + "/coords.npy") and os.path.exists(sub_path + "/AP.npy"):
            results = np.load(sub_path + "/results.npy")
            results = np.clip(results, a_min=None, a_max=-30)
            results = normalize_data(results, target_range=(-57, -30))
            # exit(1)
            coords = np.load(sub_path + "/coords.npy")
            AP = np.load(sub_path + "/AP.npy")

            self.results.append(results)
            self.coords.append(coords)
            if not self.is_test:
                if not fix:
                    for _ in range(total_iterations):
                        total_indices = results.shape[0] * results.shape[1]
                        first_index = np.random.randint(0, total_indices)
                        interval = total_indices // self.n_gateways   # Total number of indices divided by the number of sparse indices
                        sparse_indices = [first_index + i * interval for i in range(self.n_gateways)]
                        coordinates = [
                            ((index // results.shape[1]) % results.shape[0],
                             (index % results.shape[1]) % results.shape[1]) for
                            index in sparse_indices]
                        # Larger point measurement
                        self.gateway_coords.append([coords[gat_pos[0]][gat_pos[1]] for gat_pos in coordinates])

                        self.rssis.append([results[gat_pos[0]][gat_pos[1]] for gat_pos in coordinates])
                        self.AP.append(AP)
                else:
                    actual_points = int(self.n_gateways // patch_span // patch_span)

                    total_indices = results.shape[0] * results.shape[1]
                    first_index = np.random.randint(0, total_indices)
                    interval = total_indices // actual_points  # Total number of indices divided by the number of sparse indices
                    sparse_indices = [first_index + i * interval for i in range(actual_points)]

                    # Pixel-specific measurement
                    for index in sparse_indices:
                        for dia_idx_x in range(patch_span):
                            for dia_idx_y in range(patch_span):
                                x_index = min(results.shape[0] - 1, (index // results.shape[1]) % results.shape[0] + dia_idx_x)
                                y_index = min(results.shape[1] - 1, (index % results.shape[1]) % results.shape[1] + dia_idx_y)
                                coordinates.append([x_index, y_index])

                    for _ in range(total_iterations):
                        # Larger point measurement
                        self.gateway_coords.append([coords[gat_pos[0]][gat_pos[1]] for gat_pos in coordinates])

                        self.rssis.append([results[gat_pos[0]][gat_pos[1]] for gat_pos in coordinates])
                        self.AP.append(AP)

            else:
                self.n_gateways = results.shape[0] * results.shape[1]
                for i in range(results.shape[0]):
                    for j in range(results.shape[1]):
                        self.gateway_coords.append(coords[i][j])
                        self.rssis.append(results[i][j])
                self.AP.append(AP)

                # self.gateway_coords = [self.gateway_coords]
                # self.rssis = [self.rssis]
            # Test for single room
            self.results_shape = results.shape
        self.gateway_pos = torch.tensor(np.array(self.gateway_coords), dtype=torch.float32)
        # self.gateway_pos_dir = os.path.join(datadir, 'gateway_position.yml')
        # self.rssi_dir = os.path.join(datadir, 'gateway_rssi.csv')
        # self.dataset_index = np.loadtxt(indexdir, dtype=int)
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # Load transmitter position
        self.tx_poses = torch.tensor(np.array(self.AP), dtype=torch.float32)
        self.tx_poses = self.tx_poses / scale_worldsize

        # Load gateway received RSSI
        self.rssis = torch.tensor(np.array(self.rssis), dtype=torch.float32)

        if self.is_test:
            self.gateway_pos = self.gateway_pos.view(1, self.n_gateways, -1)
            self.rssis = self.rssis.view(1, -1)

        # self.coords = torch.tensor(self.coords, dtype=torch.float32)

        self.nn_inputs, self.nn_labels = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        nn_inputs : tensor. [n_samples, 978]. The inputs for training
                    tx_pos:3, ray_o:3, ray_d:9x36x3,
        nn_labels : tensor. [n_samples, 1]. The RSSI labels for training
        """
        ## NOTE! Large dataset may cause OOM?
        # print("load_data")
        # print(self.rssis.size())
        # print(self.tx_poses.size())
        # print(self.gateway_pos.size())
        # nn_inputs = torch.tensor(np.zeros((torch.sum(self.rssis != -150), 3 + 3 + 3 * self.alpha_res * self.beta_res)), dtype=torch.float32)
        # nn_labels = torch.tensor(np.zeros((torch.sum(self.rssis != -150), 1)), dtype=torch.float32)
        nn_inputs = torch.tensor(np.zeros((self.rssis.numel(), 3 + 3 + 3 * self.alpha_res * self.beta_res)),
                                 dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((self.rssis.numel(), 1)), dtype=torch.float32)

        ## Load data
        data_counter = 0
        for idx in range(len(self.rssis)):
            rssis = self.rssis[idx]
            tx_pos = self.tx_poses[idx].view(-1)  # [3]
            ## generate rays origin at gateways
            gateways_ray_o, gateways_rays_d = self.gen_rays_gateways(idx)
            for i_gateway, rssi in enumerate(rssis):
                # if rssi != -150:
                gateway_ray_o = gateways_ray_o[i_gateway].view(-1)  # [3]
                gateway_rays_d = gateways_rays_d[i_gateway].view(-1)  # [n_rays x 3]
                nn_inputs[data_counter] = torch.cat([tx_pos, gateway_ray_o, gateway_rays_d], dim=-1)
                nn_labels[data_counter] = rssi
                data_counter += 1

        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels

    def gen_rays_gateways(self, idx):
        """generate sample rays origin at gateways, for each gateways, we sample 36x9 rays

        Returns
        -------
        r_o : tensor. [n_gateways, 1, 3]. The origin of rays
        r_d : tensor. [n_gateways, n_rays, 3]. The direction of rays, unit vector
        """

        alphas = torch.linspace(0, 350, self.alpha_res) / 180 * np.pi
        betas = torch.linspace(10, 90, self.beta_res) / 180 * np.pi
        alphas = alphas.repeat(self.beta_res)  # [0,1,2,3,....]
        betas = betas.repeat_interleave(self.alpha_res)  # [0,0,0,0,...]

        radius = 1
        x = radius * torch.cos(alphas) * torch.cos(betas)  # (1*360)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        r_d = torch.stack([x, y, z], axis=0).T  # [9*36, 3]
        r_d = r_d.expand([self.n_gateways, self.beta_res * self.alpha_res, 3])  # [n_gateways, 9*36, 3]
        r_o = self.gateway_pos[idx].unsqueeze(1)  # [21, 1, 3]
        r_o, r_d = r_o.contiguous(), r_d.contiguous()

        return r_o, r_d

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]

    def __len__(self):
        rssis = self.rssis
        # return torch.sum(rssis != -150)
        return torch.numel(rssis)
