import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from math import exp

class SelfsupervisedLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 photo_weight=1,
                 smooth_weight=10,
                 ):
        super(SelfsupervisedLoss, self).__init__()
        self.photo_weight = photo_weight
        self.smooth_weight = smooth_weight

    def ssim(self, img1, img2, window_size=11):
        _, channel, h, w = img1.size()
        window = self.create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return self._ssim(img1, img2, window, window_size, channel)

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def warp(self, x, disp):
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
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map

    def compute_feat_reconstr_loss(self, warped, ref, mask, simple=True):
        if simple:
            return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
        else:
            alpha = 0.5
            ref_dx, ref_dy = self.gradient(ref * mask)
            warped_dx, warped_dy = self.gradient(warped * mask)
            photo_loss = F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
            grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
            return (1 - alpha) * photo_loss + alpha * grad_loss
    
    def feat_recon_loss(self, featL, featR, disp, mask=None):
        b, c, h, w = featL.shape

        featL = F.interpolate(featL, scale_factor=4, mode='bilinear')
        featR = F.interpolate(featR, scale_factor=4, mode='bilinear')
    
        # if scaled_disp.shape[-2] != h or scaled_disp.shape[-1] != w:
        #     # compute scale per level and scale gtDisp
        #     scale = scaled_disp.shape[-1] / (w * 1.0)
        #     scaled_disp = scaled_disp.clone() / scale
        #     scaled_disp = F.adaptive_avg_pool2d(scaled_disp, (h, w))

        recon_featL = self.warp(featR, disp)
        loss = self.compute_feat_reconstr_loss(recon_featL[:, :, :, 60:], featL[:, :, :, 60:], mask[:, :, :, 60:], simple=False) 
        return loss

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()
        img = F.interpolate(img, (h, w), mode='area')
        dis_weight = 1e-3
        cvt_weight = 1e-3

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)
        feature_dyx, feature_dyy = self.gradient(feature_dy)

        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        return -dis_weight * smooth1 + cvt_weight * smooth2
    
    def smooth_item_l0_5(self, x, beta):
        mask = x<beta
        if not mask.sum()== 0:
            x[mask] = 32768*torch.square(x[mask])
        if not (~mask).sum()== 0:
            x[~mask] = torch.sqrt(x[~mask])
        return x

    def feature_consistency_loss(self, clean_feats, adverse_feats, mask=None):
        clean_feats_norm, adverse_feats_norm = F.normalize(clean_feats, dim=1), F.normalize(adverse_feats, dim=1)
        if mask:
            cosine_similarity = F.cosine_similarity(clean_feats_norm[mask], adverse_feats_norm[mask])
            cosine_similarity = cosine_similarity.mean()
        else:
            cosine_similarity = F.cosine_similarity(clean_feats_norm, adverse_feats_norm)
            cosine_similarity = cosine_similarity.mean()
            
        return 1 - cosine_similarity + 1e-6


    def LR_Check_v1(self, dispL, dispR, thres=2.5):
        # # left to right
        disp_L2R = self.warp(dispL, -dispR)
        dispR_thres = (disp_L2R - dispR).abs()
        mask_R = dispR_thres > thres
        dispR[mask_R] = 0.

        # right to left
        disp_R2L = self.warp(dispR, dispL)
        dispL_thres = (disp_R2L - dispL).abs()
        mask_L = dispL_thres > thres

        return (~mask_L).detach()

    def loss_disp_smoothness(self, disp, img):
        img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
        weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

        loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum() + 1e-3)

        return loss

    def smooth_l0_5(self, pred, gt, beta=0.00097656):
        assert pred.shape == gt.shape, "the shapes of pred and gt are not matched."
        error = pred - gt
        abs_error = torch.abs(error)
    
        smooth_sqrt_abs_error = self.smooth_item_l0_5(abs_error, beta)
        loss = torch.mean(smooth_sqrt_abs_error)
        return loss

    def compute_reconstr_loss_l0_5(self, warped, ref, mask=None, simple=True):
        if simple:
            if mask != None:
                return self.smooth_l0_5(warped*mask, ref*mask)
            else:
                return self.smooth_l0_5(warped, ref)
        else:
            alpha = 0.5
            ref_dx, ref_dy = self.gradient(ref * mask)
            warped_dx, warped_dy = self.gradient(warped * mask)
            photo_loss = self.smooth_l0_5(warped*mask, ref*mask)
            grad_loss = self.smooth_l0_5(warped_dx, ref_dx) + \
                    self.smooth_l0_5(warped_dy, ref_dy)
            return (1 - alpha) * photo_loss + alpha * grad_loss


    def reconstruction_loss(self, img, recon_img, valid_mask=None):
        if valid_mask != None:
            return 0.15 * self.compute_reconstr_loss_l0_5(img, recon_img, valid_mask) + \
                0.85 * (valid_mask * (1 - self.ssim(img, recon_img)) / 2).mean()
        else:
            return 0.15 * self.compute_reconstr_loss_l0_5(img, recon_img) + \
                0.85 * ((1 - self.ssim(img, recon_img)) / 2).mean()

    def forward(self, left_img, right_img, left_disps, right_disps, loss_weights, bidir=False, use_mask=False):
        losses = {}
        valid_mask_L, valid_mask_R = None, None

        assert len(loss_weights) == len(left_disps)
        if use_mask:
            thres = 1.5
            with torch.no_grad():
                valid_mask_L = self.LR_Check_v1(left_disps[-1].clone().detach(), right_disps[-1].clone().detach(), thres=thres)
                if bidir:
                    valid_mask_R = self.LR_Check_v1(right_disps[-1].flip(3).clone().detach(), left_disps[-1].flip(3).clone().detach(), thres=thres).flip(3)
                
                valid_mask_L = valid_mask_L.detach()
                if bidir:
                    valid_mask_R = valid_mask_R.detach()

        photo_loss = torch.FloatTensor([0.]).cuda()

        for i in range(len(left_disps)):
            recon_left_img = self.warp(right_img, left_disps[i])
            photo_loss += loss_weights[i] * self.reconstruction_loss(left_img, recon_left_img, valid_mask_L)

        if bidir:
            left_img_ast = left_img.flip(3)
            for i in range(len(right_disps)):
                right_disp_ast = right_disps[i].flip(3)
                recon_right_img_ast = self.warp(left_img_ast, right_disp_ast)
                recon_right_img = recon_right_img_ast.flip(3)
                photo_loss += loss_weights[i] * self.reconstruction_loss(right_img, recon_right_img, valid_mask_R)

        smooth_loss = torch.FloatTensor([0.]).cuda()

        for i in range(len(left_disps)):
            smooth_loss += self.loss_disp_smoothness(left_disps[i]/20, left_img) * loss_weights[i]
            if bidir:
                smooth_loss += self.loss_disp_smoothness(right_disps[i]/20, right_img) * loss_weights[i]

        total_loss = self.smooth_weight * smooth_loss + self.photo_weight * photo_loss
        
        losses = {
            "smooth_loss": self.smooth_weight * smooth_loss.item(),
            "photo_loss": self.photo_weight * photo_loss.item(),
            "total_loss": total_loss,
            "valid_mask": valid_mask_L,
        }
        return losses
