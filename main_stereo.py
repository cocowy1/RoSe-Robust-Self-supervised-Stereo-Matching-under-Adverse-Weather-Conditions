import torch
from torch.utils.data import DataLoader
import tensorboard
import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch.nn.parallel.distributed

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataloader.stereo.datasets import build_dataset
from utils import misc

from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from utils.visualization import disp_error_img, save_images
from loss.stereo_metric import d1_metric
from evaluate_stereo import (validate_things, validate_kitti15, validate_kitti12, validate_eth3d, validate_robust_drivingstereo, validate_robust,
                             validate_middlebury, validate_drivingstereo, validate_MS2, validate_robust_MS2, inference_stereo,
                             )

from models.robust_raft import Robust_RAFT

# from models.robust_cfnet_v2 import robust_cfnet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# try:
#     from torch.cuda.amp import GradScaler
# except:
#     class GradScaler:
#         def __init__(self):
#             pass
#         def scale(self, loss):
#             return loss
#         def unscale_(self, optimizer):
#             pass
#         def step(self, optimizer):
#             optimizer.step()
#         def update(self):
#             pass

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='/data/ywang/my_projects/RoSe/weights/', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='sceneflow', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default='robust', type=str, nargs='+')
    parser.add_argument('--max_disp', default=192, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=320, type=int)
    parser.add_argument('--img_width', default=832, type=int)
    parser.add_argument('--padding_factor', default=64, type=int)
    parser.add_argument('--validate_iters', type=int, default=24, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--train_iters', type=int, default=16, help='number of flow-field updates during training forward pass')


    # training
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--bf16', default=False, type=bool)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--seed', default=326, type=int)


    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--supervised_type', default="pseudo_supervised", type=str,
                        help='optional task-specific local regression refinement')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)

    # log
    parser.add_argument('--summary_freq', default=200, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=200, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=200, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=500, type=int)
    parser.add_argument('--num_steps', default=10000, type=int)
    parser.add_argument('--save_model_path', default='/data/ywang/my_projects/RoSe/checkpoints/', type=str)
    
    # distributed training
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    #  igev Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*4, help="hidden state and context dimensions")
    parser.add_argument('--VFM_dims', nargs='+', type=int, default=[128, 128, 128, 128], help="hidden state and context dimensions")
    parser.add_argument('--mixed_precision', default=True, action='store_false', help='use mixed precision')
    parser.add_argument('--proxy', default=False, type=bool,
                        help='optional task-specific local regression refinement')
    parser.add_argument('--enhancement', default=False, type=bool,
                        help='optional feature refinement')
    parser.add_argument('--vfm_type', default='damv2', type=str,
                        help='optional task-specific local regression refinement')
    
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

    return parser


def main(args):
    print_info = not args.eval and args.inference_dir is None and \
                 args.inference_dir_left is None and args.inference_dir_right is None

    if print_info and args.local_rank == 0:
        print(args)

        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.save_model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))
        
        setup_for_distributed(args.local_rank == 0)

    # model
    print("args.bf16 is:", args.bf16)

    # model = Robust_Stereo(feature_channels=args.feature_channels,
    #                  num_scales=args.num_scales,
    #                  upsample_factor=args.upsample_factor,
    #                  num_head=args.num_head,
    #                  ffn_dim_expansion=args.ffn_dim_expansion,
    #                  reg_refine=args.reg_refine,
    #                  use_AGCL=args.use_AGCL,
    #                  num_transformer_layers=args.num_transformer_layers).to(device)

    # model = robust_pwcnet(args.max_disp).to(device)
    # model = ACVNet(args.max_disp, False, False).to(device)

    # model = robust_cfnet().to(device)

    model = Robust_RAFT(args).to(device)

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)

    # if print_info:
    #     print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 0:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print("Parameter Count: %d, learnable parameter : %d" %(num_params, sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if not args.eval and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # if args.bf16 == True:
    #         scaler = torch.cuda.amp.GradScaler()

    scaler = torch.cuda.amp.GradScaler()

    pre_scale = int(scaler.get_scale()) if args.bf16 else None

    start_epoch = 0
    start_step = 0

    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)

        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model.module.load_state_dict(checkpoint["model"], strict=False)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']

        if print_info:
            print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))


    train_data = build_dataset(args)

    print('=> {} training samples found in the training set'.format(len(train_data)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank
        )
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              sampler=train_sampler,
                              )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
            pct_start=0.02, cycle_momentum=False, anneal_strategy='linear')

    total_steps = start_step
    epoch = start_epoch
    print('=> Start training...')


    while total_steps < args.num_steps:
        model.train()

        # mannually change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.local_rank == 0:
            summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], total_steps + 1)

        for i, sample in enumerate(train_loader):
            left_ori = sample['left_ori'].to(device).to(torch.float32)  # [B, 3, H, W]
            right_ori = sample['right_ori'].to(device).to(torch.float32)
            left = sample['left'].to(device).to(torch.float32)  # [B, 3, H, W]
            right = sample['right'].to(device).to(torch.float32)
            
            gt_disp = sample['disp'].to(device).to(torch.float32) # [B, H, W]
            gt_disp = gt_disp.unsqueeze(1).detach()
            
            mask = ((gt_disp > 0) & (gt_disp < args.max_disp))

            if not mask.any():
                continue

            if args.bf16:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                    pred_results = model(left, right, iters=args.train_iters)
            else:
                pred_results = model(left, right, iters=args.train_iters) 
            pred_disps = pred_results["disp_preds"]

            disp_loss = 0
            all_loss = []

            # loss weights
            loss_weights = [0.9 ** (len(pred_disps) - 1 - power) for power in
                            range(len(pred_disps))]

            for k in range(len(pred_disps)):
                pred_disp = pred_disps[k].to(torch.float32)
                weight = loss_weights[k]
                curr_loss = F.l1_loss(pred_disp[mask], gt_disp[mask], reduction='mean')
                disp_loss += weight * curr_loss
            
            if "recon_img" in pred_results:
                recon_loss = torch.mean((left_ori - pred_results["recon_img"]) ** 2) 
                disp_loss += recon_loss

            if "loss_ckr" in pred_results:
                loss_ckr = pred_results["loss_ckr"].squeeze()
                disp_loss += loss_ckr
                
            # more efficient zero_grad
            for param in model.parameters():
                param.grad = None

            if args.bf16:
                scaler.scale(disp_loss).backward()
            else:
                disp_loss.backward()

            ### trick for gradient value is nan
            parameters = model.parameters()
            gradients = [param.grad for param in parameters if param.grad is not None]

            # 检查并替换 NaN 值
            for grad in gradients:
                if torch.isnan(grad).any():
                    grad[torch.isnan(grad)] = 0  # 将 NaN 值替换为 0 或其他合适的数值
                    # print("detect the unbounded graident, revise it to zero !")
         
   
            if args.bf16:
                scaler.unscale_(optimizer)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.99)

            if args.bf16:
                scaler.step(optimizer)
                scaler.update()
                new_scale = int(scaler.get_scale())
                if new_scale == pre_scale:  # 只有scale不变表示优化进行了
                    lr_scheduler.step()
                pre_scale = new_scale
            else:
                optimizer.step()
                lr_scheduler.step()


            total_steps += 1

            if total_steps % args.summary_freq == 0 and args.local_rank == 0:
                img_summary = dict()
                img_summary['left_ori'] = left
                img_summary['right_ori'] = right
                img_summary['gt_disp'] = gt_disp
                img_summary['pred_disp'] = pred_disps[-1]

                if "recon_img" in pred_results:
                    img_summary['recon_img'] = pred_results["recon_img"].to(torch.float32)

                img_summary['disp_error'] = disp_error_img(pred_disp.squeeze(1), gt_disp.squeeze(1))
                save_images(summary_writer, 'train', img_summary, total_steps)

                epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                print('step: %06d \t epe: %.3f' % (total_steps, epe.item()))

                summary_writer.add_scalar('train/epe', epe.item(), total_steps)
                summary_writer.add_scalar('train/disp_loss', disp_loss.item(), total_steps)

                if "recon_img" in pred_results:
                    summary_writer.add_scalar('train/recon_loss', recon_loss.item(), total_steps)

                if "loss_ckr" in pred_results:
                    summary_writer.add_scalar('train/loss_ckr', loss_ckr.item(), total_steps)
                    
                # save all losses
                for s in range(len(all_loss)):
                    save_name = 'train/loss' + str(len(all_loss) - s - 1)
                    save_value = all_loss[s]
                    summary_writer.add_scalar(save_name, save_value, total_steps)

                d1 = d1_metric(pred_disp, gt_disp, mask)
                summary_writer.add_scalar('train/d1', d1.item(), total_steps)

            # always save the latest model for resuming training
            if args.local_rank == 0 and total_steps % args.save_latest_ckpt_freq == 0:
                # Save lastest checkpoint after each epoch
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                save_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                }

                torch.save(save_dict, checkpoint_path)

            # save checkpoint of specific epoch
            if total_steps > 998 and args.local_rank == 0 and total_steps % args.save_ckpt_freq == 0:
                print('Save checkpoint at step: %d' % total_steps)
                checkpoint_path = os.path.join(args.save_model_path, 'step_%06d.pth' % total_steps)

                save_dict = {
                    'model': model_without_ddp.state_dict(),
                }

                torch.save(save_dict, checkpoint_path)

            # validation
            if total_steps > 998 and total_steps % args.val_freq == 0:
                val_results = {}

                if 'robust_drivingstereo' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_robust_drivingstereo(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_robust_drivingstereo(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )

                    if args.local_rank == 0:
                        val_results.update(results_dict)


                if 'robust' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_robust(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_robust(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )

                    if args.local_rank == 0:
                        val_results.update(results_dict)
                        

                if 'kitti' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_kitti15(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_kitti15(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'kitti' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_kitti12(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_kitti12(model_without_ddp,
                            max_disp = args.max_disp,
                            iters=args.validate_iters,      
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )

                    if args.local_rank == 0:
                        val_results.update(results_dict)


                # if 'drivingstereo' in args.val_dataset:
                #     if args.bf16:
                #         with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                #             results_dict = validate_drivingstereo(model_without_ddp,
                #             max_disp=args.max_disp,                          
                #             padding_factor=args.padding_factor,
                #             inference_size=args.inference_size,
                #             attn_type=args.attn_type,
                #             attn_splits_list=args.attn_splits_list,
                #             corr_radius_list=args.corr_radius_list,
                #             prop_radius_list=args.prop_radius_list,
                #             count_time=args.count_time,
                #         )
                #     else:
                #         results_dict = validate_drivingstereo(model_without_ddp,
                #             max_disp=args.max_disp,   
                #             padding_factor=args.padding_factor,
                #             inference_size=args.inference_size,
                #             attn_type=args.attn_type,
                #             attn_splits_list=args.attn_splits_list,
                #             corr_radius_list=args.corr_radius_list,
                #             prop_radius_list=args.prop_radius_list,
                #             count_time=args.count_time,
                #         )
                
                #     if args.local_rank == 0:
                #         val_results.update(results_dict)


                if 'MS2' in args.val_dataset:
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_MS2(model_without_ddp,
                            max_disp=args.max_disp,                          
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_MS2(model_without_ddp,
                            max_disp=args.max_disp,   
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                
                    if args.local_rank == 0:
                        val_results.update(results_dict)


                if 'robust_MS2' in args.val_dataset:
                    model.eval()
                    if args.bf16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
                            results_dict = validate_robust_MS2(model_without_ddp,
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )
                    else:
                        results_dict = validate_robust_MS2(model_without_ddp,
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            count_time=args.count_time,
                        )

                    if args.local_rank == 0:
                        val_results.update(results_dict)

                model.train()

                if args.local_rank == 0:
                    # save to tensorboard
                    for key in val_results:
                        tag = key.split('_')[0]
                        tag = tag + '/' + key
                        summary_writer.add_scalar(tag, val_results[key], total_steps)

                    # save validation results to file
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)
                        # order of metrics
                        metrics = ['things_epe', 'things_d1',
                                'kitti15_epe', 'kitti15_d1', 'kitti15_1px', 'kitti15_3px',
                                'kitti12_epe', 'kitti12_d1', 'kitti12_1px', 'kitti12_3px',
                                'drivingstereo_epe', 'drivingstereo_d1', 'drivingstereo_1px', 'drivingstereo_3px',
                                'MS2_epe', 'MS2_d1', 'MS2_1px', 'MS2_3px',
                                'eth3d_epe', 'eth3d_1px',
                                'middlebury_epe', 'middlebury_2px',
                                 'clear_drivingstereo_epe', 'clear_drivingstereo_d1', 'clear_drivingstereo_1px', 'clear_drivingstereo_3px',
                                'night_drivingstereo_epe', 'night_drivingstereo_d1', 'night_drivingstereo_1px', 'night_drivingstereo_3px',
                                'foggy_drivingstereo_epe', 'foggy_drivingstereo_d1', 'foggy_drivingstereo_1px', 'foggy_drivingstereo_3px',
                                'rainy_drivingstereo_epe', 'rainy_drivingstereo_d1', 'rainy_drivingstereo_1px', 'rainy_drivingstereo_3px',
                                'clear_MS2_epe', 'clear_MS2_d1', 'clear_MS2_1px', 'clear_MS2_3px',
                                'night_MS2_epe', 'night_MS2_d1', 'night_MS2_1px', 'night_MS2_3px',
                                'foggy_MS2_epe', 'foggy_MS2_d1', 'foggy_MS2_1px', 'foggy_MS2_3px',
                                'rainy_MS2_epe', 'rainy_MS2_d1', 'rainy_MS2_1px', 'rainy_MS2_3px',
                            ]


                        eval_metrics = []
                        for metric in metrics:
                            if metric in val_results.keys():
                                eval_metrics.append(metric)

                        metrics_values = [val_results[metric] for metric in eval_metrics]

                        num_metrics = len(eval_metrics)

                        f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                        f.write(("| {:20.4f} " * num_metrics).format(*metrics_values))

                        f.write('\n\n')

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
