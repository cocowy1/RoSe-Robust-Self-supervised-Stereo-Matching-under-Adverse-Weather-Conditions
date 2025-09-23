import torch
from models.unimatch_dtd.utils import BackprojectDepth
from models.unimatch_dtd.utils import Project3D
import torch.nn.functional as F
import pdb





class ImageWarping(torch.nn.Module):
    def __init__(self, batch_size, image_height, image_width):
        super(ImageWarping, self).__init__()
        self.to_pts = BackprojectDepth(batch_size, image_height, image_width).cuda()
        self.to_pix = Project3D(batch_size, image_height, image_width).cuda()

    def forward(self, image, depth, pose, intrinsics):

        assert (
            len(image.shape) == 4
            and len(depth.shape) == 4
            and len(pose.shape) == 3
            and len(intrinsics.shape) == 3
        )

        pts = self.to_pts(depth, intrinsics)
        pixels = self.to_pix(pts, intrinsics, pose)

        reconstrcted_image = F.grid_sample(image, pixels, padding_mode="border")

        return reconstrcted_image





class SingleScaleImageWarping(torch.nn.Module):
    def __init__(self, batch_size, image_width, image_height, num_scales):
        super(SingleScaleImageWarping, self).__init__()
        self.num_scales = num_scales

        self.image_warper = ImageWarping(batch_size, image_width, image_height).cuda()
        print("Single Scale ImageWarping initialized")

    def forward(self, images, depths, poses, intrinsics):

        assert len(images) == len(depths) == len(intrinsics) == self.num_scales


        reconstrcted_images = []
        for i in range(self.num_scales):
            reconstrcted_images.append(
                self.image_warper(images[0], depths[i], poses, intrinsics[0])
            )

        return reconstrcted_images


class MultiScaleImageWarping(torch.nn.Module):
    def __init__(self, batch_size, image_width, image_height, num_scales):
        super(MultiScaleImageWarping, self).__init__()
        self.num_scales = num_scales
        self.image_warpers = []
        for i in range(num_scales):
            scaled_image_width = image_width // (2**i)
            scaled_image_height = image_height // (2**i)

            self.image_warpers.append(
                ImageWarping(batch_size, scaled_image_width, scaled_image_height).cuda()
            )
        print("MultiScaleImageWarping initialized")

    def forward(self, images, depths, poses, intrinsics):

        assert len(images) == len(depths) == len(intrinsics) == self.num_scales


        reconstrcted_images = []
        for i in range(self.num_scales):
            reconstrcted_images.append(
                self.image_warpers[i](images[i], depths[i], poses, intrinsics[i])
            )

        return reconstrcted_images



class MaskedImageWarping(torch.nn.Module):

    def __init__(self, batch_size, image_width, image_height):
        super(MaskedImageWarping, self).__init__()
        self.to_pts = BackprojectDepth(batch_size, image_height, image_width).cuda()
        self.to_pix = Project3D(batch_size, image_height, image_width).cuda()

        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size

    def generate_mask(self, scene_coords, pix_coords):

        # in this function we are going to generate a mask 

        # scene coords 

        pdb.set_trace() 
        delta = torch.norm(pix_coords[:, 2, :, :], self.to_pts.get_original_pix_coords()[:, 2, :, :], dim=1,keepdim=True)
        scene_coords_mask = torch.logical_and(scene_coords[:, 2, :, :] > 0, scene_coords[:, 2, :, :] < 100)
        pix_coords_mask = delta > 32 # 32 is the threshold for the flow
        mask = torch.logical_and(scene_coords_mask, pix_coords_mask)

        




    def forward(self, image, depth, pose, intrinsics):

        assert (
            len(image.shape) == 4
            and len(depth.shape) == 4
            and len(pose.shape) == 3
            and len(intrinsics.shape) == 3
        )

        pts = self.to_pts(depth, intrinsics)

        scene_coords = torch.matmul(pose, pts)
        pixels = self.to_pix(pts, intrinsics, pose)
        pix_coords = self.to_pix.get_original_pix_coords()

        mask = self.generate_mask(scene_coords, pix_coords)

        reconstrcted_image = F.grid_sample(image, pixels, padding_mode="border")

        return reconstrcted_image
