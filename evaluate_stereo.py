import time
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from tqdm import tqdm 
from imageio import imread, imsave
import cv2
from PIL import Image
from glob import glob

from utils.visualization import disp_error_img, save_images

from loss.stereo_metric import d1_metric, thres_metric
from dataloader.stereo.datasets import (Driving, DrivingStereo, RobustDrivingStereo, MS2, RobustMS2,
                                        FlyingThings3D, KITTI15, KITTI12, ETH3DStereo, MiddleburyEval3)
from dataloader.stereo import transforms


from utils.utils import InputPadder

from utils.file_io import write_pfm
from utils.visualization import vis_disparity

from models.raft_stereo import RAFTStereo
from models.robust_raft import Robust_RAFT

# from models.robust_raft_codebook import Robust_RAFT

# from models.cfnet import robust_cfnet
# from models.cfnet import robust_cfnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def create_kitti_submission(model,
                            max_disp=192, 
                            output_path='disp_0',
                            padding_factor=32,
                            inference_size=None,
                            iters=24,
                            ):
    """ create submission for the KITTI leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = KITTI12(mode='testing', transform=val_transform)

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in enumerate(test_dataset):
        left = sample['left_ori'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right_ori'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)


        if i > 10:
            torch.cuda.synchronize()
            time_start = time.perf_counter()
            
            
        pred_disp = model(left, right, 
                          iters=iters, 
                          test_mode=True
                          )['disp_preds'][-1]  # [1, H, W]

        if i > 10:
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - time_start
            print('this is %d-th sample, inference tims:%.3f'%(i, inference_time))

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        save_name = os.path.join(output_path, left_name)

        imsave(save_name, (pred_disp.squeeze().cpu().numpy() * 256.).astype(np.uint16))


@torch.no_grad()
def create_eth3d_submission(model,
                            output_path='output',
                            padding_factor=16,
                            attn_type=None,
                            attn_splits_list=False,
                            corr_radius_list=False,
                            prop_radius_list=False,
                            inference_size=None,
                            submission_mode='train',
                            save_vis_disp=False,
                            ):
    """ create submission for the eth3d stereo leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = ETH3DStereo(mode=submission_mode,
                               transform=val_transform,
                               save_filename=True
                               )

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fixed_inference_size = inference_size

    for i, sample in enumerate(test_dataset):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        nearest_size = [int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(left.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = left.shape[-2:]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(left, right,
                      attn_type=attn_type,
                      attn_splits_list=attn_splits_list,
                      corr_radius_list=corr_radius_list,
                      prop_radius_list=prop_radius_list,
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(left, right,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          )['disp_preds'][-1]  # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))

        if save_vis_disp:
            save_name = os.path.join(output_path, filename + '.png')
            disp = vis_disparity(pred_disp.cpu().numpy())
            cv2.imwrite(save_name, disp)
        else:
            save_disp_name = os.path.join(output_path, filename + '.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(output_path, filename + '.txt')
            with open(save_runtime_name, 'w') as f:
                f.write('runtime ' + str(inference_time))


@torch.no_grad()
def create_middlebury_submission(model,
                                 output_path='output',
                                 padding_factor=16,
                                 attn_type=None,
                                 attn_splits_list=False,
                                 corr_radius_list=False,
                                 prop_radius_list=False,
                                 inference_size=None,
                                 submission_mode='train',
                                 save_vis_disp=False,
                                 ):
    """ create submission for the Middlebury leaderboard """
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    test_dataset = MiddleburyEval3(mode=submission_mode,
                                   resolution='F',
                                   transform=val_transform,
                                   save_filename=True,
                                   )

    num_samples = len(test_dataset)
    print('Number of test samples: %d' % num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, sample in enumerate(test_dataset):
        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        left_name = sample['left_name']

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        # warpup to measure inference time
        if i == 0:
            for _ in range(5):
                model(left, right,
                      attn_type=attn_type,
                      attn_splits_list=attn_splits_list,
                      corr_radius_list=corr_radius_list,
                      prop_radius_list=prop_radius_list,
                      )

        torch.cuda.synchronize()
        time_start = time.perf_counter()

        pred_disp = model(left, right,
                          attn_type=attn_type,
                          attn_splits_list=attn_splits_list,
                          corr_radius_list=corr_radius_list,
                          prop_radius_list=prop_radius_list,
                          task='stereo',
                          )['flow_preds'][-1]  # [1, H, W]

        torch.cuda.synchronize()
        inference_time = time.perf_counter() - time_start

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        filename = os.path.basename(os.path.dirname(left_name))  # works for both windows and linux

        if save_vis_disp:
            save_name = os.path.join(output_path, filename + '.png')
            disp = vis_disparity(pred_disp.cpu().numpy())
            cv2.imwrite(save_name, disp)
        else:
            save_disp_dir = os.path.join(output_path, filename)
            os.makedirs(save_disp_dir, exist_ok=True)

            save_disp_name = os.path.join(save_disp_dir, 'disp0GMStereo.pfm')
            # save disp
            write_pfm(save_disp_name, pred_disp.cpu().numpy())
            # save runtime
            save_runtime_name = os.path.join(save_disp_dir, 'timeGMStereo.txt')
            with open(save_runtime_name, 'w') as f:
                f.write(str(inference_time))


@torch.no_grad()
def validate_things(model,
                    max_disp=400,
                    padding_factor=16,
                    inference_size=None,
                    attn_type=None,
                    num_iters_per_scale=None,
                    attn_splits_list=None,
                    corr_radius_list=None,
                    prop_radius_list=None,
                    ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = FlyingThings3D(mode='TEST', transform=val_transform)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 1000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              num_iters_per_scale=num_iters_per_scale,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)

        val_epe += epe.item()
        val_d1 += d1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples

    print('Validation things EPE: %.3f, D1: %.4f' % (
        mean_epe, mean_d1))

    results['things_epe'] = mean_epe
    results['things_d1'] = mean_d1

    return results



@torch.no_grad()
def validate_kitti12(model,
                     max_disp=192,
                     padding_factor=16,
                     inference_size=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = KITTI12(transform=val_transform,
                          )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()

        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation KITTI12 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['kitti12_epe'] = mean_epe
    results['kitti12_d1'] = mean_d1
    results['kitti12_3px'] = mean_thres3
    results['kitti12_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results



@torch.no_grad()
def validate_kitti15(model,
                     max_disp=192,
                     padding_factor=16,
                     inference_size=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = KITTI15(transform=val_transform,
                          )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()

        mask = (gt_disp > 0) & (gt_disp < max_disp)

        if not mask.any():
            continue

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation KITTI15 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['kitti15_epe'] = mean_epe
    results['kitti15_d1'] = mean_d1
    results['kitti15_3px'] = mean_thres3
    results['kitti15_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results



@torch.no_grad()
def validate_drivingstereo(model,
                     max_disp=128,
                     padding_factor=64,
                     num_reg_refine=1,
                     data_path=None,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = DrivingStereo(transform=val_transform,
                                mode="testing",)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 5000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0
        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              max_disp=max_disp,
                              num_reg_refine=num_reg_refine,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation drivingstereo EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['drivingstereo_epe'] = mean_epe
    results['drivingstereo_d1'] = mean_d1
    results['drivingstereo_3px'] = mean_thres3
    results['drivingstereo_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results


@torch.no_grad()
def validate_robust_drivingstereo(model,
                     max_disp=128,
                     iters=16,
                     padding_factor=32,
                     inference_size=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    data_path = '/data/ywang/dataset/DrivingStereo/DrivingStereo/adverse_weather/'
    # weathers = ["clear", "cloudy", "foggy", "rainy"]
    weathers = ["clear", "foggy", "rainy"]
    
    for weather in weathers:
        val_dataset = RobustDrivingStereo(data_dir=data_path+weather,
                                transform=val_transform, mode="testing")
    
        num_samples = len(val_dataset)
        print('=> %d samples found in the validation %s weather set' % (num_samples, weather))

        val_epe = 0
        val_d1 = 0
        val_thres3 = 0
        val_thres1 = 0

        if count_time:
            total_time = 0
            num_runs = 100

        valid_samples = 0

        for i, sample in tqdm(enumerate(val_dataset)):
            if debug and i > 10:
                break

            if i % 5000 == 0:
                print('=> Validating %d/%d' % (i, num_samples))

            left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

            if inference_size is None:
                padder = InputPadder(left.shape, divis_by=padding_factor)
                left, right = padder.pad(left, right)
            else:
                ori_size = left.shape[-2:]
                left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
                right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

            mask = (gt_disp > 0) & (gt_disp < max_disp)
            if not mask.any():
                continue

            valid_samples += 1

            if count_time and i >= 5:
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            with torch.no_grad():
                pred_disp = model(image1=left, image2=right, iters=iters)['disp_preds'][-1]  # [1, H, W]

            if count_time and i >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if i >= num_runs + 4:
                    break
                
            # remove padding
            if inference_size is None:
                pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
            else:
                # resize back
                pred_disp = F.interpolate(pred_disp, size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            pred_disp = pred_disp.squeeze()
            gt_disp = gt_disp.squeeze()
            mask = (gt_disp > 0)
            
            # import matplotlib.pyplot as plt
            # disp_data_path = '/data/ywang/my_projects/RoSe/mid_weather_output/' + str(weather)  + '/' + str(i) + '_disp.svg'
            # plt.imsave(disp_data_path, pred_disp.squeeze().cpu().numpy(), cmap='jet')

            # disp_data_path = '/data/ywang/my_projects/RoSe/mid_weather_output/' + str(weather)  + '/' + str(i) + '_gt.svg'
            # plt.imsave(disp_data_path, gt_disp.squeeze().cpu().numpy(), cmap='jet')
            
            # error_data_path = '/data/ywang/my_projects/RoSe/mid_weather_output/' + str(weather) + '/' + str(i) + '_error.svg'
            # error_map = disp_error_img(pred_disp.unsqueeze(0), gt_disp.unsqueeze(0))
            # plt.imsave(error_data_path, error_map.permute(0,2,3,1).squeeze().cpu().numpy(), cmap='jet')
            
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres3 += thres3.item()
            val_thres1 += thres1.item()

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples * 100
        mean_thres3 = val_thres3 / valid_samples * 100
        mean_thres1 = val_thres1 / valid_samples * 100

        print('Validation robust drivingstereo %s weather subset EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        weather, mean_epe, mean_d1, mean_thres1, mean_thres3))

        results.update({str(weather)+'_drivingstereo_epe': mean_epe})
        results.update({str(weather)+'_drivingstereo_d1': mean_d1})
        results.update({str(weather)+'_drivingstereo_3px': mean_thres3})
        results.update({str(weather)+'_drivingstereo_1px': mean_thres1})


    # data_path = '/data/ywang/dataset/MS2/adverse_weather/'
    # weathers = ["night"]

    # for weather in weathers:
    #     val_dataset = RobustMS2(data_dir=data_path + weather,
    #                             transform=val_transform, mode="testing")
    
    #     num_samples = len(val_dataset)
    #     print('=> %d samples found in the validation %s weather set' % (num_samples, weather))

    #     val_epe = 0
    #     val_d1 = 0
    #     val_thres3 = 0
    #     val_thres1 = 0

    #     if count_time:
    #         total_time = 0
    #         num_runs = 100

    #     valid_samples = 0

    #     for i, sample in tqdm(enumerate(val_dataset)):
    #         if debug and i > 10:
    #             break

    #         if i % 5000 == 0:
    #             print('=> Validating %d/%d' % (i, num_samples))

    #         left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
    #         right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
    #         gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

    #         if inference_size is None:
    #             padder = InputPadder(left.shape, padding_factor=padding_factor)
    #             left, right = padder.pad(left, right)
    #         else:
    #             ori_size = left.shape[-2:]
    #             left = F.interpolate(left, size=inference_size, mode='bilinear',
    #                              align_corners=True)
    #             right = F.interpolate(right, size=inference_size, mode='bilinear',
    #                               align_corners=True)

    #         mask = gt_disp > 0
    #         if not mask.any():
    #             continue

    #         valid_samples += 1

    #         if count_time and i >= 5:
    #             torch.cuda.synchronize()
    #             time_start = time.perf_counter()

    #         with torch.no_grad():
    #             pred_disp = model(left=left, right=right,
    #                           attn_type=attn_type,
    #                           max_disp=max_disp,
    #                           num_reg_refine=num_reg_refine,
    #                           attn_splits_list=attn_splits_list,
    #                           corr_radius_list=corr_radius_list,
    #                           prop_radius_list=prop_radius_list,
    #                           )['disp_preds'][-1]  # [1, H, W]

    #         if count_time and i >= 5:
    #             torch.cuda.synchronize()
    #             total_time += time.perf_counter() - time_start

    #             if i >= num_runs + 4:
    #                 break

    #         # remove padding
    #         if inference_size is None:
    #             pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
    #         else:
    #             # resize back
    #             pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
    #                                   align_corners=True).squeeze(1)[0]  # [H, W]
    #             pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    #         pred_disp = pred_disp.squeeze()
    #         gt_disp = gt_disp.squeeze()
    #         mask = (gt_disp > 0)

    #         epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
    #         d1 = d1_metric(pred_disp, gt_disp, mask)
    #         thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
    #         thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

    #         val_epe += epe.item()
    #         val_d1 += d1.item()
    #         val_thres3 += thres3.item()
    #         val_thres1 += thres1.item()

    #     mean_epe = val_epe / valid_samples
    #     mean_d1 = val_d1 / valid_samples * 100
    #     mean_thres3 = val_thres3 / valid_samples * 100
    #     mean_thres1 = val_thres1 / valid_samples * 100

    #     print('Validation robust MS2 %s weather subset EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
    #     weather, mean_epe, mean_d1, mean_thres1, mean_thres3))

    #     results.update({str(weather)+'_MS2_epe': mean_epe})
    #     results.update({str(weather)+'_MS2_d1': mean_d1})
    #     results.update({str(weather)+'_MS2_3px': mean_thres3})
    #     results.update({str(weather)+'_MS2_1px': mean_thres1})

    # if count_time:
    #     print('Time: %.6fs' % (total_time / num_runs))

    return results




@torch.no_grad()
def validate_robust(model,
                     max_disp=192,
                     iters=24,
                     padding_factor=32,
                     inference_size=None,
                     count_time=True,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    # val_transform_list = None
    
    val_transform = transforms.Compose(val_transform_list)

    val_dataset_12 = KITTI12(transform=val_transform,)
    
    val_dataset_15 = KITTI15(transform=val_transform,)
        
    val_dataset_mid = MiddleburyEval3(transform=val_transform,)

    val_dataset_eth3d = ETH3DStereo(transform=val_transform,)

    val_datasets = {"eth3d": val_dataset_eth3d, "kit12": val_dataset_12, "kit15": val_dataset_15, "mid": val_dataset_mid}
    # val_datasets = {"kit12": val_dataset_12, "kit15": val_dataset_15,}

    for name, val_dataset in val_datasets.items():
        num_samples = len(val_dataset)
        print('=> %d samples found in the validation %s dataset' %(num_samples, name))

        val_epe = 0
        val_d1 = 0
        val_thres3 = 0
        val_thres2 = 0
        val_thres1 = 0

        valid_samples = 0
        elapse_counter = 0
        elapse_time = 0
        avgtime = 0
        
        for i, sample in tqdm(enumerate(val_dataset)):

            left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            gt_disp = sample['disp'].to(device).to(torch.float32).unsqueeze(0)  # [H, W]
                
            if inference_size is None:
                padder = InputPadder(left.shape, divis_by=padding_factor)
                left, right = padder.pad(left, right)
            else:
                ori_size = left.shape[-2:]
                left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
                right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

            valid_samples += 1

            if i > 10:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                elapse_counter += 1
                
            with torch.no_grad():
                outputs = model(image1=left, image2=right,
                              iters=iters, test_mode=True
                              ) # [1, H, W]
            
            if i > 10:
                elapse_time += time.perf_counter() - start_time
          
            pred_disp = outputs['disp_preds'][-1] 
            
                   
            # remove padding
            if inference_size is None:
                pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
            else:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            pred_disp = pred_disp.squeeze()
            gt_disp = gt_disp.squeeze()
            mask = (gt_disp > 0.05) & (gt_disp < max_disp)
                
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres3 += thres3.item()
            val_thres2 += thres2.item()
            val_thres1 += thres1.item()
        
        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples * 100
        mean_thres3 = val_thres3 / valid_samples * 100
        mean_thres2 = val_thres2 / valid_samples * 100
        mean_thres1 = val_thres1 / valid_samples * 100
        avgtime = elapse_time / elapse_counter
        
        print('Validation %s dataset EPE: %.3f, D1: %.4f, 1px: %.4f,  2px: %.4f, 3px: %.4f, avgtime: %.4f' % (
            name, mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3, avgtime))

        results[str(name) + '_epe'] = mean_epe
        results[str(name) + '_d1'] = mean_d1
        results[str(name) + '_3px'] = mean_thres3
        results[str(name) + '_2px'] = mean_thres2
        results[str(name) + '_1px'] = mean_thres1
        results[str(name) + '_avgtime'] = avgtime
    return results


@torch.no_grad()
def validate_MS2(model,
                     max_disp=128,
                     padding_factor=32,
                     num_reg_refine=1,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MS2(transform=val_transform,
                                mode="testing",)

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres3 = 0
    val_thres1 = 0

    if count_time:
        total_time = 0
        num_runs = 100

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if debug and i > 10:
            break

        if i % 5000 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0
        if not mask.any():
            continue

        valid_samples += 1

        if count_time and i >= 5:
            torch.cuda.synchronize()
            time_start = time.perf_counter()

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              max_disp=max_disp,
                              num_reg_refine=num_reg_refine,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        if count_time and i >= 5:
            torch.cuda.synchronize()
            total_time += time.perf_counter() - time_start

            if i >= num_runs + 4:
                break

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        pred_disp = pred_disp.squeeze()
        gt_disp = gt_disp.squeeze()
        mask = gt_disp > 0
        
        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres3 += thres3.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples * 100
    mean_thres3 = val_thres3 / valid_samples * 100
    mean_thres1 = val_thres1 / valid_samples * 100

    print('Validation MS2 EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        mean_epe, mean_d1, mean_thres1, mean_thres3))

    results['MS2_epe'] = mean_epe
    results['MS2_d1'] = mean_d1
    results['MS2_3px'] = mean_thres3
    results['MS2_1px'] = mean_thres1

    if count_time:
        print('Time: %.6fs' % (total_time / num_runs))

    return results



@torch.no_grad()
def validate_robust_MS2(model,
                     max_disp=128,
                     padding_factor=32,
                     num_reg_refine=1,
                     inference_size=None,
                     attn_type=None,
                     attn_splits_list=None,
                     corr_radius_list=None,
                     prop_radius_list=None,
                     count_time=False,
                     debug=False,
                     ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    data_path = '/data/ywang/dataset/MS2/adverse_weather/'
    weathers = ["night"]

    for weather in weathers:
        val_dataset = RobustMS2(data_dir=data_path+weather,
                                transform=val_transform, mode="testing")

        num_samples = len(val_dataset)
        print('=> %d samples found in the validation %s weather set' % (num_samples, weather))

        val_epe = 0
        val_d1 = 0
        val_thres3 = 0
        val_thres1 = 0

        if count_time:
            total_time = 0
            num_runs = 100

        valid_samples = 0

        for i, sample in enumerate(val_dataset):
            if debug and i > 10:
                break

            if i % 5000 == 0:
                print('=> Validating %d/%d' % (i, num_samples))

            left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
            gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

            if inference_size is None:
                padder = InputPadder(left.shape, padding_factor=padding_factor)
                left, right = padder.pad(left, right)
            else:
                ori_size = left.shape[-2:]
                left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
                right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

            mask = gt_disp > 0
            if not mask.any():
                continue

            valid_samples += 1

            if count_time and i >= 5:
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            with torch.no_grad():
                pred_disp = model(left, right,
                              attn_type=attn_type,
                              max_disp=max_disp,
                              num_reg_refine=num_reg_refine,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

            if count_time and i >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if i >= num_runs + 4:
                    break

            # remove padding
            if inference_size is None:
                pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
            else:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            pred_disp = pred_disp.squeeze()
            gt_disp = gt_disp.squeeze()
            mask = gt_disp > 0
        
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres3 += thres3.item()
            val_thres1 += thres1.item()

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples * 100
        mean_thres3 = val_thres3 / valid_samples * 100
        mean_thres1 = val_thres1 / valid_samples * 100

        print('Validation robust MS2 %s weather subset EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
        weather, mean_epe, mean_d1, mean_thres1, mean_thres3))

        results.update({str(weather)+'_MS2_epe': mean_epe})
        results.update({str(weather)+'_MS2_d1': mean_d1})
        results.update({str(weather)+'_MS2_3px': mean_thres3})
        results.update({str(weather)+'_MS2_1px': mean_thres1})

    # data_path = '/data/ywang/dataset/DrivingStereo/DrivingStereo/adverse_weather/'
    # weathers = ["foggy"]

    # for weather in weathers:
    #     val_dataset = RobustDrivingStereo(data_dir=data_path + weather,
    #                             transform=val_transform, mode="testing")
    
    #     num_samples = len(val_dataset)
    #     print('=> %d samples found in the validation %s weather set' % (num_samples, weather))

    #     val_epe = 0
    #     val_d1 = 0
    #     val_thres3 = 0
    #     val_thres1 = 0

    #     if count_time:
    #         total_time = 0
    #         num_runs = 100

    #     valid_samples = 0

    #     for i, sample in enumerate(val_dataset):
    #         if debug and i > 10:
    #             break

    #         if i % 5000 == 0:
    #             print('=> Validating %d/%d' % (i, num_samples))

    #         left = sample['left_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
    #         right = sample['right_ori'].to(device).to(torch.float32).unsqueeze(0)  # [1, 3, H, W]
    #         gt_disp = sample['disp'].to(device).to(torch.float32)  # [H, W]

    #         if inference_size is None:
    #             padder = InputPadder(left.shape, padding_factor=padding_factor)
    #             left, right = padder.pad(left, right)
    #         else:
    #             ori_size = left.shape[-2:]
    #             left = F.interpolate(left, size=inference_size, mode='bilinear',
    #                              align_corners=True)
    #             right = F.interpolate(right, size=inference_size, mode='bilinear',
    #                               align_corners=True)

    #         mask = gt_disp > 0
    #         if not mask.any():
    #             continue

    #         valid_samples += 1

    #         if count_time and i >= 5:
    #             torch.cuda.synchronize()
    #             time_start = time.perf_counter()

    #         with torch.no_grad():
    #             pred_disp = model(left, right,
    #                           attn_type=attn_type,
    #                           max_disp=max_disp,
    #                           num_reg_refine=num_reg_refine,
    #                           attn_splits_list=attn_splits_list,
    #                           corr_radius_list=corr_radius_list,
    #                           prop_radius_list=prop_radius_list,
    #                           )['disp_preds'][-1]  # [1, H, W]

    #         if count_time and i >= 5:
    #             torch.cuda.synchronize()
    #             total_time += time.perf_counter() - time_start

    #             if i >= num_runs + 4:
    #                 break

    #         # remove padding
    #         if inference_size is None:
    #             pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
    #         else:
    #             # resize back
    #             pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
    #                                   align_corners=True).squeeze(1)[0]  # [H, W]
    #             pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    #         pred_disp = pred_disp.squeeze()
    #         gt_disp = gt_disp.squeeze()
    #         mask = (gt_disp > 0)

    #         epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
    #         d1 = d1_metric(pred_disp, gt_disp, mask)
    #         thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
    #         thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

    #         val_epe += epe.item()
    #         val_d1 += d1.item()
    #         val_thres3 += thres3.item()
    #         val_thres1 += thres1.item()

    #     mean_epe = val_epe / valid_samples
    #     mean_d1 = val_d1 / valid_samples * 100
    #     mean_thres3 = val_thres3 / valid_samples * 100
    #     mean_thres1 = val_thres1 / valid_samples * 100

    #     print('Validation robust drivingstereo %s weather subset EPE: %.3f, D1: %.4f, 1px: %.4f, 3px: %.4f' % (
    #     weather, mean_epe, mean_d1, mean_thres1, mean_thres3))

    #     results.update({str(weather)+'_drivingstereo_epe': mean_epe})
    #     results.update({str(weather)+'_drivingstereo_d1': mean_d1})
    #     results.update({str(weather)+'_drivingstereo_3px': mean_thres3})
    #     results.update({str(weather)+'_drivingstereo_1px': mean_thres1})

    return results



@torch.no_grad()
def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        vgrid = grid.cuda()
    else:
        vgrid = grid
    
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
        
    return output



@torch.no_grad()
def LR_Check_v1(dispL, dispR, thres=2.1):
    # # left to right
    disp_L2R = warp(dispL, -dispR)
    dispR_thres = (disp_L2R - dispR).abs()
    mask_R = dispR_thres > thres
    dispR[mask_R] = 0.

    # right to left
    disp_R2L = warp(dispR, dispL)
    dispL_thres = (disp_R2L - dispL).abs()
    mask_L = dispL_thres > thres

    return (~mask_L).detach()
    


@torch.no_grad()
def validate_eth3d(model,
                   padding_factor=16,
                   inference_size=None,
                   attn_type=None,
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = ETH3DStereo(transform=val_transform,
                              )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres1 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()

    mean_epe = val_epe / valid_samples
    mean_thres1 = val_thres1 / valid_samples

    print('Validation ETH3D EPE: %.3f, 1px: %.4f' % (
        mean_epe, mean_thres1))

    results['eth3d_epe'] = mean_epe
    results['eth3d_1px'] = mean_thres1

    return results


@torch.no_grad()
def validate_middlebury(model,
                        padding_factor=16,
                        inference_size=None,
                        attn_type=None,
                        attn_splits_list=None,
                        corr_radius_list=None,
                        prop_radius_list=None,
                        resolution='H',
                        ):
    model.eval()
    results = {}

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    val_dataset = MiddleburyEval3(transform=val_transform,
                                  resolution=resolution,
                                  )

    num_samples = len(val_dataset)
    print('=> %d samples found in the validation set' % num_samples)

    val_epe = 0
    val_d1 = 0
    val_thres2 = 0

    valid_samples = 0

    for i, sample in enumerate(val_dataset):
        if i % 100 == 0:
            print('=> Validating %d/%d' % (i, num_samples))

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]
        gt_disp = sample['disp'].to(device)  # [H, W]

        if inference_size is None:
            padder = InputPadder(left.shape, padding_factor=padding_factor)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        mask = gt_disp > 0

        if not mask.any():
            continue

        valid_samples += 1

        with torch.no_grad():
            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              )['disp_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size is None:
            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]
        else:
            # resize back
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres2 += thres2.item()

    mean_epe = val_epe / valid_samples
    mean_thres2 = val_thres2 / valid_samples

    print('Validation Middlebury EPE: %.3f, 2px: %.4f' % (
        mean_epe, mean_thres2))

    results['middlebury_epe'] = mean_epe
    results['middlebury_2px'] = mean_thres2

    return results


@torch.no_grad()
def inference_stereo(model,
                    max_disp=192,
                     iters=24,
                     output_path='output',
                     padding_factor=64,
                     inference_size=None,
                     save_pfm_disp=False,
                     ):
    model.eval()

    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    left_name = '/data/ywang/dataset/MS2/adverse_weather/night/_2021-08-13-22-16-02/img_left/000265.png'
    right_name = '/data/ywang/dataset/MS2/adverse_weather/night/_2021-08-13-22-16-02/img_right/000265.png'
    gt_disp = '/data/ywang/dataset/MS2/adverse_weather/night/_2021-08-13-22-16-02/disp_filtered/000265.png'

    left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
    right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)
    gt_disp = np.array(Image.open(gt_disp))
    gt_disp = gt_disp.astype(np.float32) / 256.

    sample = {'left_ori': left, 'right_ori': right, 'disp': gt_disp}
    sample = val_transform(sample)

    left = sample['left_ori'].to(device).unsqueeze(0)  # [1, 3, H, W]
    right = sample['right_ori'].to(device).unsqueeze(0)  # [1, 3, H, W]
    gt_disp = sample['disp'].to(device).unsqueeze(0)  # [1, 3, H, W]

    if inference_size is None:
        padder = InputPadder(left.shape, padding_factor=padding_factor)
        left, right = padder.pad(left, right)
    else:
        ori_size = left.shape[-2:]
        left = F.interpolate(left, size=inference_size, mode='bilinear',
                                 align_corners=True)
        right = F.interpolate(right, size=inference_size, mode='bilinear',
                                  align_corners=True)
        
    with torch.no_grad():
        pred_disp = model(image1=left, image2=right,
                              iters=iters, test_mode=True)['disp_preds'][-1]  # [1, H, W]

          
    # remove padding
    if inference_size is None:
        pred_disp = padder.unpad(pred_disp.unsqueeze(0).unsqueeze(0))[0]  # [H, W]
    else:
        # resize back
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='bilinear',
                                      align_corners=True).squeeze(1)[0]  # [H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])


    pred_disp = pred_disp.squeeze()
    gt_disp = gt_disp.squeeze()
         
    import matplotlib.pyplot as plt
    disp_data_path = '/data/ywang/my_projects/RoSe/night_ablation/mid/265_disp.svg'
    plt.imsave(disp_data_path, pred_disp.squeeze().cpu().numpy(), vmax=gt_disp.max(), cmap='jet')

    disp_data_path = '/data/ywang/my_projects/RoSe/night_ablation/mid/265_gt.svg'
    plt.imsave(disp_data_path, gt_disp.squeeze().cpu().numpy(), vmax=gt_disp.max(), cmap='jet')
            
    error_data_path = '/data/ywang/my_projects/RoSe/night_ablation/mid/265_error.svg'
    error_map = disp_error_img(pred_disp.unsqueeze(0), gt_disp.unsqueeze(0))
    plt.imsave(error_data_path, error_map.permute(0,2,3,1).squeeze().cpu().numpy(), cmap='jet')
      

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', default='robust_drivingstereo', type=str, nargs='+')
    parser.add_argument('--max_disp', default=192, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=320, type=int)
    parser.add_argument('--img_width', default=832, type=int)
    parser.add_argument('--padding_factor', default=64, type=int)
    parser.add_argument('--validate_iters', type=int, default=24, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--train_iters', type=int, default=16, help='number of flow-field updates during training forward pass')


    # resume pretrained model or resume training
    parser.add_argument('--resume', default='/data/ywang/my_projects/RoSe_ori/checkpoints/robust_raft/ds/step_006200(final_ds).pth', type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_false',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_false')
    parser.add_argument('--resume_exclude_upsampler', action='store_true')
    parser.add_argument('--reg_refine', default=True, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--use_AGCL', default=True, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--code_book', default=True, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--use_IGEV', default=False, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--supervised_type', default="supervised", type=str,
                        help='optional task-specific local regression refinement')
    
    #model:  igev Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*4, help="hidden state and context dimensions")
    parser.add_argument('--VFM_dims', nargs='+', type=int, default=[128, 128, 128, 128], help="hidden state and context dimensions")
    parser.add_argument('--proxy', default=False, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--vfm_type', default='damv2', type=str,
                        help='optional task-specific local regression refinement')

    # evaluation
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='F', choices=['Q', 'H', 'F'])
    
    # distributed training
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_bidir_disp', action='store_true',
                        help='predict both left and right disparities')
    parser.add_argument('--pred_right_disp', action='store_true',
                        help='predict right disparity')
    parser.add_argument('--save_pfm_disp', action='store_true',
                        help='save predicted disparity as .pfm format')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    #### RAFT Initialize
    # model = RAFTStereo(args).to(device)
    # model = torch.nn.DataParallel(model)
    # checkpoint = torch.load(args.resume)
    # model.module.load_state_dict(checkpoint["model"], strict=True)


    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model.to(device),
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank)
    #     model_without_ddp = model.module
    # else:
    #     if torch.cuda.device_count() > 1:
    #         print('Use %d GPUs' % torch.cuda.device_count())
    #         model = torch.nn.DataParallel(model)

    #         model_without_ddp = model.module
    #     else:
    #         model_without_ddp = model

    # #### robust RAFT Initialize
    model = Robust_RAFT(args).to(device)
    model = torch.nn.DataParallel(model)
    
    # model = robust_cfnet(max_disp=args.max_disp).to(device)
    # model = torch.nn.DataParallel(model)
    
    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model.module.load_state_dict(checkpoint["model"], strict=True)

    model.cuda()
    model.eval()

    print(f"The model has {format(sum(p.numel() for p in model.parameters())/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, max_disp=args.max_disp, 
                            iters=args.validate_iters,                      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)


    elif args.dataset == 'robust':
        validate_robust(model, padding_factor=args.padding_factor,
                         inference_size=args.inference_size, iters=args.validate_iters)
        
        
    elif args.dataset == 'kitti':
        validate_kitti15(model, max_disp=args.max_disp,    
                            iters=args.validate_iters,                     
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)
        validate_kitti12(model, max_disp=args.max_disp,    
                            iters=args.validate_iters,                     
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)
        
    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, max_disp=args.max_disp,     
                            iters=args.validate_iters,                  
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)

    elif args.dataset == 'drivingstereo':
        validate_drivingstereo(model, max_disp=args.max_disp,
                            iters=args.validate_iters,                   
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)

    elif args.dataset == 'robust_drivingstereo':
        validate_robust_drivingstereo(model, max_disp=args.max_disp,  
                            iters=args.validate_iters,                            
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)

    if args.dataset == 'kit15':
        create_kitti_submission(model, max_disp=args.max_disp, 
                            iters=args.validate_iters,                      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)

    if args.dataset == 'night':
        inference_stereo(model, max_disp=args.max_disp, 
                            iters=args.validate_iters,                      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size)
