import torch
from .masked_stereo import DTD
from torchvision import transforms as T
from .dino_v1 import ViTExtractor as DinoV1ExtractFeatures

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dtd_config(args):
    args.num_scales = 2
    args.feature_channels = 128
    args.upsample_factor = 4
    args.num_head = 1
    args.ffn_dim_expansion = 4
    args.num_transformer_layers = 6
    args.attn_type = "self_swin2d_cross_1d"
    args.attn_splits_list = [1, 8]
    args.corr_radius_list = [-1, 4]
    args.prop_radius_list = [-1, 1]
    args.num_reg_refine = 3
    args.mask_thr = 0.2

    return args

def get_dino_config(args):

    args.dino_model_name= 'dino_vits8'
    args.dino_patch_size = 8
    args.dino_stride = 4
    args.dino_input_dim = 384
    args.dino_layers = [5,11]

    return args

class StereoDepthNet(torch.nn.Module):
    def __init__(self, args, reg_refine=False):
        super(StereoDepthNet, self).__init__()
        args = get_dtd_config(args)
        args = get_dino_config(args)
        if reg_refine:
            args.reg_refine = reg_refine
            print("=> reg refine is set to: ", args.reg_refine)

        dtd_model = DTD(
            feature_channels=args.feature_channels,
            num_scales=args.num_scales,
            upsample_factor=args.upsample_factor,
            num_head=args.num_head,
            ffn_dim_expansion=args.ffn_dim_expansion,
            num_transformer_layers=args.num_transformer_layers,
            input_dim = args.dino_input_dim,
            mask_thr=args.mask_thr
        )

        model = torch.nn.DataParallel(dtd_model)
        self.model = model.module
        self.args = args
        self.image_net_normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.dino = DinoV1ExtractFeatures(args.dino_model_name, args.dino_stride, device="cuda")
        self.dino.model.eval()
        self.model.extract_feature = self.extract_features_dino_v1
        self.feat = []


    def extract_features_dino_v1(self, image0, image1):

        image_shape = image0.shape[2:]
        fine_height = 1 + (image_shape[0] - self.args.dino_patch_size) // self.args.dino_stride
        fine_width = 1 + (image_shape[1] - self.args.dino_patch_size) // self.args.dino_stride

        coarse_height = image_shape[0] // self.args.dino_patch_size
        coarse_width = image_shape[1] // self.args.dino_patch_size


        with torch.no_grad():
            desc0 = self.dino._extract_features(image0, 
                                                layers=self.args.dino_layers, 
                                                facet="token")
            desc1 = self.dino._extract_features(image1, 
                                                layers=self.args.dino_layers, 
                                                facet="token")

            coarse_feat0 = desc0[-1][:, 1:, :].permute(0, 2, 1)
            coarse_feat1 = desc1[-1][:, 1:, :].permute(0, 2, 1)

            fine_feat0 = desc0[0][:, 1:, :].permute(0, 2, 1)
            fine_feat1 = desc1[0][:, 1:, :].permute(0, 2, 1)

        batch_size = desc0[0].shape[0]
        coarse_feat0 = coarse_feat0.view(batch_size, -1, coarse_height, coarse_width)
        coarse_feat1 = coarse_feat1.contiguous().view(batch_size, -1, coarse_height, coarse_width)

        fine_feat0 = fine_feat0.contiguous().view(batch_size, -1, fine_height, fine_width)
        fine_feat1 = fine_feat1.contiguous().view(batch_size, -1, fine_height, fine_width)

        fine_feat0 = torch.nn.functional.pad(fine_feat0, (0, 1, 0, 1))
        fine_feat1 = torch.nn.functional.pad(fine_feat1, (0, 1, 0, 1))

        self.feat = ([coarse_feat0, fine_feat0],[coarse_feat1, fine_feat1]) 

        return [coarse_feat0, fine_feat0], [coarse_feat1, fine_feat1]


    def forward(self, left, right, return_distance=False, masks=False, norm=False, **kwargs):

        if norm:
            left = self.image_net_normalizer(left)
            right = self.image_net_normalizer(right)

        outputs = self.model(
            left,
            right,
            attn_type=self.args.attn_type,
            attn_splits_list=self.args.attn_splits_list,
            corr_radius_list=self.args.corr_radius_list,
            prop_radius_list=self.args.prop_radius_list,
        )

        pred_disp = outputs["disp_preds"]
        if return_distance:
            distance = outputs["nn_distance"]
            if masks:
                masks = outputs["bad_pixel_mask"]
                return outputs
            return outputs

        return outputs
