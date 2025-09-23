import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unimatch_dtd.transformer import FeatureTransformer
from models.unimatch_dtd.matching import global_correlation_softmax_stereo, local_correlation_softmax_stereo
from models.unimatch_dtd.attention import SelfAttnPropagation
from models.unimatch_dtd.geometry import flow_warp as disp_warp
from models.unimatch_dtd.utils import feature_add_position, upsample_flow_with_mask

class DTD(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 input_dim = 384,
                 output_dim = 128,
                 mask_thr = 0.2
                 ):
        super(DTD, self).__init__()
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.mask_thr = mask_thr
        # CNN
        self.backbone = None
        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        self.projector = nn.Sequential(nn.Conv2d(input_dim, 256, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, output_dim, 1))

        # convex upsampling simiar to RAFT
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    
    def extract_feature(self, img0, img1):
        # this should be from DINO
        return None, None

    def upsample_disp(self, disp, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_disp = F.interpolate(disp, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((disp, feature), dim=1)
            mask = self.upsampler(concat)
            up_disp = upsample_flow_with_mask(disp, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_disp

    def get_mask(self, features):

        '''
        This function is compute the mask the features that are not unique enought to have 
        the distance grather than a threshold

        Args:
            feature: the feature map of the image
            thr: the threshold of the distance

        Returns:
            mask: the mask of the feature map
    
        '''
        normlize_features = torch.nn.functional.normalize(features, dim=1)
        re_feat = normlize_features.permute(0, 2, 3, 1).contiguous()
        re_feat = re_feat.view(re_feat.shape[0], -1, re_feat.shape[3])
        similarity = torch.bmm(re_feat, re_feat.permute(0, 2, 1).contiguous())
        nearest_nn_inds = torch.topk(similarity, k=2, dim=-1, largest=True)[1][:, :, 1]
        nearest_nn = torch.gather(re_feat, dim=1, index=nearest_nn_inds.unsqueeze(-1).expand(-1, -1, re_feat.shape[-1]))
        distance = torch.norm(re_feat - nearest_nn, dim=-1)
        mask = torch.where(distance > self.mask_thr, torch.ones_like(distance), torch.zeros_like(distance))
        mask = mask.view(features.shape[0],1, features.shape[2], features.shape[3])

        return mask, distance


    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                **kwargs
                ):


        results_dict = {}
        pred_disparities = []
        distances = []
        masks = []
        disp = None
        with torch.no_grad():
            # list of features, resolution low to high
            feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx].detach(), feature1_list[scale_idx].detach()
            feature0 = self.projector(feature0) # to downsample the feature
            feature1 = self.projector(feature1) # to downsample the feature
            bad_pixel_mask, nearest_neighbour_distances = self.get_mask(feature0)
            distances.append(nearest_neighbour_distances)
            masks.append(bad_pixel_mask)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))
            if scale_idx > 0:
                disp = F.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=True) * 2
            if disp is not None:
                disp = disp.detach()
                zeros = torch.zeros_like(disp)  # [B, 1, H, W]
                # NOTE: reverse disp, disparity is positive
                displace = torch.cat((-disp, zeros), dim=1)  # [B, 2, H, W]
                feature1 = disp_warp(feature1, displace)  # [B, C, H, W]


            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # here they do global attention, can we do local attention?            
            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )

                      
            if corr_radius == -1:  # global matching
                disp_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
            else: # local matching
                disp_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]

                
            disp_pred = disp_pred * bad_pixel_mask
            # flow or residual flow
            disp = disp + disp_pred if disp is not None else disp_pred
            disp = disp.clamp(min=0)  # positive disparity
            # upsample to the original resolution for supervison at training time only
            if self.training:
                disp_bilinear = self.upsample_disp(disp, None, 
                                                   bilinear=True,
                                                    upsample_factor=upsample_factor,
                                                   is_depth=False)# this last flag is redundant but added to use unimatch code
                
                pred_disparities.append(disp_bilinear)

            disp = self.feature_flow_attn(feature0, disp.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )
            # bilinear exclude the last one and for the last one use RAFT- style upsampling
            if self.training and scale_idx < self.num_scales - 1:
                disp_up = self.upsample_disp(disp, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=False)
                pred_disparities.append(disp_up)

            if scale_idx == self.num_scales - 1:
                disp_pad = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
                disp_up_pad = self.upsample_disp(disp_pad, feature0)
                disp_up = -disp_up_pad[:, :1]  # [B, 1, H, W]
                pred_disparities.append(disp_up)
                
        for i in range(len(pred_disparities)):
            pred_disparities[i] = pred_disparities[i].squeeze(1)  # [B, H, W]
        results_dict.update({'disp_preds': pred_disparities})
        results_dict.update({'nn_distance': distances})
        results_dict.update({'bad_pixel_mask': masks})

        return results_dict
