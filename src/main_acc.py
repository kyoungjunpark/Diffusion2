import torch
from config.config import args as args_config
import time
import random
import os
import transformers

os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = args_config.port

import diffusers as diffusers
from accelerate import Accelerator



from utils.point_loader import collation_fn_frames, collation_fn_eval_all


import json
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utility
from model import get as get_model
from data import get as get_data
from loss import get as get_loss
from summary import get as get_summary
from metric import get as get_metric

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from torch.cuda.amp import autocast

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Minimize randomness
torch.manual_seed(args_config.seed)
np.random.seed(args_config.seed)
random.seed(args_config.seed)

torch.cuda.manual_seed_all(args_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            if args.force_maxdepth:
                new_args.max_depth = args.max_depth

    return new_args


def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None:
            model.to(device, dtype=weight_dtype)


def train(gpu, accelerator, args):
    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=gpu)
    # torch.cuda.set_device(gpu)
    logging.info("Load Train")

    logging.info(accelerator.state)
    accelerate_set_verbose(accelerator)

    # Prepare dataset
    data = get_data(args)
    logging.info("Prepared Data")

    weight_dtype = is_mixed_precision(accelerator)

    data_train = data(args, 'train')
    data_val = data(args, 'val')
    data_test = data(args, 'test')
    batch_size = args.batch_size // args.num_gpus # * args.video_fps
    logging.info("batch_size :" + str(batch_size))

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size,
        drop_last=True, collate_fn=collation_fn_frames)
    loader_val = DataLoader(
        dataset=data_val, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True,
        drop_last=False, collate_fn=collation_fn_frames)
    loader_test = DataLoader(dataset=data_test, batch_size=batch_size,
                             shuffle=False, num_workers=args.num_threads, pin_memory=True, drop_last=False,
                             collate_fn=collation_fn_frames)

    logging.info("DataLoader")

    # Network
    model = get_model(args)
    net = model(args)

    # net.cuda(gpu)

    if accelerator.is_main_process:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            # net.load_state_dict(checkpoint['net'])
            model_dict = net.state_dict()

            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict
                               and "time_video_embedding" not in k and "spatio_temp_block" not in k}

            print('Load network parameters from : {}'.format(args.pretrain))
            print("Updated parameters: ", len(pretrained_dict.keys()))

            # Update the state dict
            model_dict.update(pretrained_dict)

            # Load the updated state dict
            net.load_state_dict(model_dict)

    # Loss
    loss = get_loss(args)
    loss = loss(args)
    # loss.cuda(gpu)

    # Optimizer
    # optimizer, scheduler = utility.make_optimizer_scheduler(args, net)
    if args.split_backbone_training:
        optimizer, scheduler = utility.make_optimizer_scheduler_split(args, net)
    else:
        optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

    # net = apex.parallel.convert_syncbn_model(net)
    # net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)

    if accelerator.is_main_process:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    # amp.load_state_dict(checkpoint['amp'])

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    cast_to_gpu_and_type([net], accelerator.device, weight_dtype)

    net, optimizer, loader_train, scheduler = accelerator.prepare(
        net,
        optimizer,
        loader_train,
        scheduler,
    )
    # net = DDP(net)

    # Ensure requires_grad is True (by default it is)
    for param in net.parameters():
        param.requires_grad = True

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    if accelerator.is_main_process:
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
            os.makedirs(args.save_dir + '/test', exist_ok=True)
        except OSError:
            pass

    if accelerator.is_main_process:
        writer_train = summary(args.save_dir, 'train', args,
                               loss.loss_name, metric.metric_name)
        writer_val = summary(args.save_dir, 'val', args,
                             loss.loss_name, metric.metric_name)
        writer_test = summary(args.save_dir, 'test', args,
                              loss.loss_name, metric.metric_name)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train) + 1.0

    for epoch in range(1, args.epochs + 1):
        # Train
        net.train()
        torch.set_grad_enabled(True)

        # sampler_train.set_epoch(epoch)

        if accelerator.is_main_process:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                print(g['lr'])
                list_lr.append(g['lr'])

            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))
        num_sample = len(loader_train) * batch_size * args.num_gpus # args.video_fps

        if accelerator.is_main_process:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for _, sample in enumerate(loader_train):
            sample = {key: val for key, val in sample.items()
                      if val is not None}
            """
            # debug:
            if batch > 5:
                break
            for key, val in sample.items():
                print('the key {}'.format(key))
                print(val.shape)
            """

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    # print(param_group)
                    lr_warm_up = param_group['initial_lr'] \
                                 * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()
            if args.opt_level == 'O0':
                output = net(sample)

            else:
                with autocast():
                    output = net(sample)
            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum
            loss_val = loss_val

            with torch.autograd.set_detect_anomaly(True):
                # with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                    # scaled_loss.backward()
                accelerator.backward(loss_sum)

            optimizer.step()

            if accelerator.is_main_process:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str,
                                                              list_lr)

                pbar.set_description(error_str)
                pbar.update(batch_size * args.num_gpus)

        if accelerator.is_main_process:
            pbar.close()

            writer_train.update(epoch, sample, output)

            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.module.state_dict(),
                    'args': args
                }

            # torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))
            accelerator.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

        loader_val = accelerator.prepare(loader_val)

        # Val
        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(loader_val) * batch_size * args.num_gpus

        if accelerator.is_main_process:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val for key, val in sample.items()
                      if val is not None}
            # sample["coords"] = sample["coords"][sample["inds_recons"], :]
            if args.opt_level == 'O0':
                output = net(sample)
            else:
                with autocast():
                    output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum
            loss_val = loss_val

            if accelerator.is_main_process:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Val', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(batch_size * args.num_gpus)

        if accelerator.is_main_process:
            pbar.close()

            writer_val.update(epoch, sample, output)
            print('')

            writer_val.save(epoch, batch, sample, output)

        loader_test = accelerator.prepare(loader_test)

        ### inline test
        num_sample = len(loader_test) * batch_size * args.num_gpus

        if accelerator.is_main_process:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_test):
            sample = {key: test for key, test in sample.items()
                      if test is not None}
            """
            # debug:
            if batch > 5:
                break
            for key, test in sample.items():
                print('the key {}'.format(key))
                print(test.shape)
            """

            if args.opt_level == 'O0':
                output = net(sample)
            else:
                with autocast():
                    output = net(sample)

            loss_sum, loss_test = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum
            loss_val = loss_test

            if accelerator.is_main_process:
                metric_test = metric.evaluate(sample, output, 'train')
                writer_test.add(loss_val, metric_test)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'test', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(batch_size * args.num_gpus)

        if accelerator.is_main_process:
            pbar.close()

            writer_test.update(epoch, sample, output)
            print('')

            writer_test.save(epoch, batch, sample, output)

        torch.set_grad_enabled(True)

        scheduler.step()
        logging.info("Finish Train setup")


def test(gpu, accelerator, args):
    # Baseline
    # NOTE : the worker with gpu=0 will do logging
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=gpu)
    # torch.cuda.set_device(gpu)
    batch_size = args.batch_size // args.num_gpus  # * args.video_fps

    if args.model_name == "MRI_":
        from model.mri_model import MRI_Model
        mri = MRI_Model(args)
    elif args.model_name == "NeRF2_":

        from model.nerf2_model_real import NeRF2_Runner
        nerf = NeRF2_Runner(args)
    else:
        # Prepare dataset
        data = get_data(args)

        data_test = data(args, 'test')

        if 'front3d' in args.data_name.lower():
            sampler_test = DistributedSampler(
                data_test, num_replicas=args.num_gpus, rank=gpu, shuffle=False)
            loader_test = DataLoader(dataset=data_test, batch_size=batch_size,
                                     shuffle=False, num_workers=args.num_threads, pin_memory=True, drop_last=False,
                                     collate_fn=collation_fn_frames)
        else:
            loader_test = DataLoader(dataset=data_test, batch_size=1,
                                     shuffle=False, num_workers=args.num_threads)

        # Network
        model = get_model(args)
        net = model(args)
        # net.cuda(gpu)

        # Loss
        loss = get_loss(args)
        loss = loss(args)
        # loss.cuda(gpu)

        if accelerator.is_main_process:
            if args.pretrain is not None:
                assert os.path.exists(args.pretrain), \
                    "file not found: {}".format(args.pretrain)

                checkpoint = torch.load(args.pretrain)

        # net = nn.DataParallel(net)
        if args.split_backbone_training:
            optimizer, scheduler = utility.make_optimizer_scheduler_split(args, net)
        else:
            optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

        # et = apex.parallel.convert_syncbn_model(net)
        # net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)
        if accelerator.is_main_process:
            if args.pretrain is not None:
                assert os.path.exists(args.pretrain), \
                    "file not found: {}".format(args.pretrain)

                checkpoint = torch.load(args.pretrain)
                # net.load_state_dict(checkpoint['net'])
                model_dict = net.state_dict()

                # Filter out unnecessary keys
                pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict
                                   and "time_video_embedding" not in k and "spatio_temp_block" not in k}

                print('Load network parameters from : {}'.format(args.pretrain))
                print("Updated parameters: ", len(pretrained_dict.keys()))

                # Update the state dict
                model_dict.update(pretrained_dict)

                # Load the updated state dict
                net.load_state_dict(model_dict)

        # net = DDP(net)

        weight_dtype = is_mixed_precision(accelerator)
        cast_to_gpu_and_type([net], accelerator.device, weight_dtype)

        net, optimizer, loader_test, scheduler = accelerator.prepare(
            net,
            optimizer,
            loader_test,
            scheduler,
        )

        metric = get_metric(args)
        metric = metric(args)
        summary = get_summary(args)

        if accelerator.is_main_process:
            try:
                os.makedirs(args.save_dir, exist_ok=True)
                os.makedirs(args.save_dir + '/test', exist_ok=True)
            except OSError:
                pass

            writer_test = summary(args.save_dir, 'test', args, None, metric.metric_name)

        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(loader_test) * batch_size * args.num_gpus

        if accelerator.is_main_process:
            pbar = tqdm(total=num_sample)

        t_total = 0
        avg_t = 0
        for batch, sample in enumerate(loader_test):
            sample = {key: val for key, val in sample.items()
                      if val is not None}

            t0 = time.time()

            if args.opt_level == 'O0':
                with torch.no_grad():
                    output = net(sample)
            else:
                with autocast():
                    output = net(sample)
            loss_sum, loss_test = loss(sample, output)

            # Divide by batch size

            t1 = time.time()

            t_total += (t1 - t0)
            if accelerator.is_main_process:
                if not args.dir_real:
                    metric_val = metric.evaluate(sample, output, 'test')
                else:
                    metric_val = metric.evaluate_real(args.dir_real, args.real_name, output)

                writer_test.add(loss_test, metric_val)

                # Save data for analysis
                if args.save_image:
                    writer_test.save(args.epochs, batch, sample, output)

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{} | Test'.format(current_time)
                pbar.set_description(error_str)
                pbar.update(args.num_gpus)

        if accelerator.is_main_process:
            pbar.close()

            writer_test.update(args.epochs, sample, output)

        t_avg = t_total / num_sample
        print('Elapsed time : {} sec, '
              'Average processing time : {} sec'.format(t_total, t_avg))


def main(args):
    # Make one log on every process with the configuration for debugging.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.logger_type,
        project_dir=args.output_dir
    )

    if not args.test_only:
        # if args.no_multiprocessing:
        train(0, accelerator, args)

    else:
        test(0, accelerator, args)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        # diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        # diffusers.utils.logging.set_verbosity_error()


def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype


if __name__ == '__main__':

    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':', getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)
