import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2
import random
from PIL import Image
from utils.file_io import read_img, read_disp

from . import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataset(Dataset):
    def __init__(self,
                 transform=None,
                 is_vkitti2=False,
                 is_sintel=False,
                 is_drivingstereo=False,
                 is_MS2=False,
                 is_booster=False,
                 is_middlebury_eth3d=False,
                 is_tartanair=False,
                 is_instereo2k=False,
                 is_crestereo=False,
                 is_fallingthings=False,
                 is_raw_disp_png=False,
                 half_resolution=False,
                 quater_resolution=False,
                 supervised_type="supervised",
                 ):

        super(StereoDataset, self).__init__()

        self.transform = transform
        self.save_filename = False
        self.is_vkitti2 = is_vkitti2
        self.is_drivingstereo = is_drivingstereo
        self.is_MS2 = is_MS2
        self.is_sintel = is_sintel
        self.is_middlebury_eth3d = is_middlebury_eth3d
        self.is_tartanair = is_tartanair
        self.is_instereo2k = is_instereo2k
        self.is_crestereo = is_crestereo
        self.is_booster = is_booster
        self.is_fallingthings = is_fallingthings
        self.half_resolution = half_resolution
        self.quater_resolution = quater_resolution
        self.is_raw_disp_png = is_raw_disp_png
        self.supervised_type = supervised_type
        self.samples = []

    def __getitem__(self, index):
        sample = {}

        # file path
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        if self.supervised_type == "pseudo_supervised" or self.supervised_type == "self_supervised":
            if 'left' in sample_path.keys():
                robust_filenames = ["clear", "night", "foggy", "rainy"]
                selected_filename = random.choices(robust_filenames)
                sample_path['left'] = sample_path['left'].replace("clear", selected_filename[0])
                sample_path['right'] = sample_path['right'].replace("clear", selected_filename[0])

                sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
                sample['right'] = read_img(sample_path['right'])
        
            sample['left_ori'] = read_img(sample_path['left_ori'])  # [H, W, 3]
            sample['right_ori'] = read_img(sample_path['right_ori'])

        else:
            sample['left_ori'] = read_img(sample_path['left_ori'])  # [H, W, 3]
            sample['right_ori'] = read_img(sample_path['right_ori'])
            sample['left'] = read_img(sample_path['left_ori'])  # [H, W, 3]
            sample['right'] = read_img(sample_path['right_ori'])
            
        if 'disp' in sample_path and sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'],
                                        vkitti2=self.is_vkitti2,
                                        sintel=self.is_sintel,
                                        drivingstereo=self.is_drivingstereo,
                                        MS2=self.is_MS2,
                                        tartanair=self.is_tartanair,
                                        instereo2k=self.is_instereo2k,
                                        fallingthings=self.is_fallingthings,
                                        crestereo=self.is_crestereo,
                                        raw_disp_png=self.is_raw_disp_png,
                                       )  # [H, W]

        if 'pseudo_disp' in sample_path and sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'],
                                        vkitti2=self.is_vkitti2,
                                        sintel=self.is_sintel,
                                        drivingstereo=self.is_drivingstereo,
                                        MS2=self.is_MS2,
                                        tartanair=self.is_tartanair,
                                        instereo2k=self.is_instereo2k,
                                        fallingthings=self.is_fallingthings,
                                        crestereo=self.is_crestereo,
                                        raw_disp_png=self.is_raw_disp_png,
                                       )  # [H, W]
            
            # for middlebury and eth3d datasets, invalid is denoted as inf
            if self.is_middlebury_eth3d or self.is_crestereo:
                sample['disp'][sample['disp'] == np.inf] = 0
                occ_mask = Image.open(sample_path['disp'].replace('disp0GT.pfm', 'mask0nocc.png'))
                occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)
                sample['disp'][occ_mask != 255] = 0

        if self.half_resolution:
            sample['left'] = cv2.resize(sample['left'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            sample['right'] = cv2.resize(sample['right'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            sample['left_ori'] = cv2.resize(sample['left_ori'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            sample['right_ori'] = cv2.resize(sample['right_ori'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            sample['disp'] = cv2.resize(sample['disp'], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) * 0.5

        if self.quater_resolution:
            sample['left'] = cv2.resize(sample['left'], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            sample['right'] = cv2.resize(sample['right'], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            sample['left_ori'] = cv2.resize(sample['left_ori'], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            sample['right_ori'] = cv2.resize(sample['right_ori'], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

            sample['disp'] = cv2.resize(sample['disp'], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR) * 0.25


        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples

        return self


class FlyingThings3D(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/SceneFlow/FlyingThings3D',
                 mode='TRAIN',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(FlyingThings3D, self).__init__(transform=transform,
                                             )

        # samples: train: 22390, test: 4370
        left_files = sorted(glob(data_dir + '/' + split + '/' + mode + '/*/*/left/*.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')

            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class Monkaa(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/SceneFlow/Monkaa',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(Monkaa, self).__init__(transform=transform)

        # samples: 8664
        left_files = sorted(glob(data_dir + '/' + split + '/*/left/*.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class Driving(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/SceneFlow/Driving',
                 split='frames_finalpass',
                 transform=None,
                 ):
        super(Driving, self).__init__(transform=transform)

        # samples: 4400
        left_files = sorted(glob(data_dir + '/' + split + '/*/*/*/left/*.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            
            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'

            self.samples.append(sample)


class KITTI15(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/kitti_2015',
                 mode='training',
                 transform=None,
                 save_filename=False,
                 ):
        super(KITTI15, self).__init__(transform=transform)

        assert mode in ['training', 'testing']
        
        if mode == 'testing':
            self.save_filename = True

        # samples: train: 200
        left_files = sorted(glob(data_dir + '/' + mode + '/image_2/*_10.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('image_2', 'image_3')

            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('image_2', 'image_3')
            
            if mode != 'testing':
                sample['pseudo_disp'] = left_name.replace('image_2', 'disp_occ_0')
                sample['disp'] = left_name.replace('image_2', 'disp_occ_0')
                
            if mode == 'testing' or self.save_filename:
                # save filename
                sample['left_name'] = os.path.basename(left_name)

            self.samples.append(sample)


class KITTI12(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/kitti_2012',
                 mode='training',
                 transform=None,
                 save_filename=False,
                 ):
        super(KITTI12, self).__init__(transform=transform)

        assert mode in ['training', 'testing']

        if mode == 'testing':
            self.save_filename = True

        # samples: train: 195
        left_files = sorted(glob(data_dir + '/' + mode + '/colored_0/*_10.png'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('colored_0', 'colored_1')

            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('colored_0', 'colored_1')
            
            if mode != 'testing':
                sample['pseudo_disp'] = left_name.replace('colored_0', 'round2_disp')
                sample['disp'] = left_name.replace('colored_0', 'disp_occ')

            if mode == 'testing':
                # save filename
                sample['left_name'] = os.path.basename(left_name)

            self.samples.append(sample)


class VKITTI2(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/VKITTI2',
                 transform=None,
                 ):
        super(VKITTI2, self).__init__(transform=transform,
                                      is_vkitti2=True,
                                      )

        # total: 21260
        left_files = sorted(glob(data_dir + '/Scene*/*/frames/rgb/Camera_0/rgb*.jpg'))

        for left_name in left_files:
            sample = dict()

            sample['left_ori'] = left_name
            sample['right_ori'] = left_name.replace('/Camera_0/', '/Camera_1/')
            sample['left'] = left_name
            sample['right'] = left_name.replace('/Camera_0/', '/Camera_1/')
            
            sample['disp'] = left_name.replace('/rgb/', '/depth/').replace('rgb_', 'depth_')[:-3] + 'png'
            # sample['pseudo_disp'] = left_name.replace('/rgb/', '/depth/').replace('rgb_', 'depth_')[:-3] + 'png'

            self.samples.append(sample)


class DrivingStereo(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/DrivingStereo/DrivingStereo/',
                 filelist='/data/ywang/my_projects/RoSe/filelists/drivingstereo/',
                 transform=None,
                 mode='training',
                 supervised_type="self_supervised",
                 ):
        super(DrivingStereo, self).__init__(transform=transform, is_drivingstereo=True,
                                            supervised_type=supervised_type)

        assert mode in ['training', 'testing']

        if mode != "testing":
            data_list_file = filelist + 'train_list.txt'
        
        elif mode == "testing":
            data_list_file = filelist + 'test_list.txt'
        
        seq_data_list = [seq_name[:-1] for seq_name in open(data_list_file)] 

        # # total: 172770
        # if mode != 'testing':
        #     left_files = sorted(glob(data_dir + '/train-left-image/*/*.jpg'))
        #     right_files = sorted(glob(data_dir + '/train-right-image/*/*.jpg'))
        #     disp_files = sorted(glob(data_dir + '/train-disparity-map/*/*.png'))

        # if mode == 'testing':
        #     left_files = sorted(glob(data_dir + '/test-left-image/*/*.jpg'))
        #     right_files = sorted(glob(data_dir + '/test-right-image/*/*.jpg'))
        #     disp_files = sorted(glob(data_dir + '/test-disparity-map/*/*.png'))

        left_files = []
        right_files = []
        disp_files = []

        for seq in seq_data_list: # iterate over each sequence
            if mode != "testing":
                left_files += sorted(glob(os.path.join(data_dir + 'train-left-image/', seq, '*.jpg')))
                right_files += sorted(glob(os.path.join(data_dir + 'train-right-image/', seq, '*.jpg')))
                disp_files += sorted(glob(os.path.join(data_dir + 'train-disparity-map/', seq, '*.png')))

                # left_files += sorted(glob(os.path.join(data_dir + 'test-left-image/left-image-half-size/', seq, '*.jpg')))
                # right_files += sorted(glob(os.path.join(data_dir + 'test-right-image/right-image-half-size/', seq, '*.jpg')))
                # disp_files += sorted(glob(os.path.join(data_dir + 'test-disparity-map/disparity-map-half-size/', seq, '*.png')))
            
            if mode == "testing":
                left_files += sorted(glob(os.path.join(data_dir + 'test-left-image/left-image-half-size/', seq, '*.jpg')))
                right_files += sorted(glob(os.path.join(data_dir + 'test-right-image/right-image-half-size/', seq, '*.jpg')))
                disp_files += sorted(glob(os.path.join(data_dir + 'test-disparity-map/disparity-map-half-size/', seq, '*.png')))
            
        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            
            sample['disp'] = disp_files[i]

            self.samples.append(sample)



class RobustDrivingStereo(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/DrivingStereo/DrivingStereo/robust_train',
                 filelist=None,
                 transform=None,
                 mode='training',
                 supervised_type="pseudo_supervised",
                 ):
        super(RobustDrivingStereo, self).__init__(transform=transform, is_drivingstereo=True,
                                            supervised_type=supervised_type)

        assert mode in ['training', 'testing']

        left_files = []
        right_files = []
        disp_files = []

        # # total: 172770
        if mode != 'testing':
            left_files = sorted(glob(data_dir + '/clear/train-left-image/*/*.jpg'))
            right_files = sorted(glob(data_dir + '/clear/train-right-image/*/*.jpg'))
            if supervised_type == "pseudo_supervised":
                disp_files = sorted(glob(data_dir + '/pseudo_disparity/*/*.png'))
            else:
                disp_files = sorted(glob(data_dir + '/disparity_map/*/*.png'))

        if mode == 'testing':
            left_files = sorted(glob(data_dir + '/left-image-half-size/*.jpg'))
            right_files = sorted(glob(data_dir + '/right-image-half-size/*.jpg'))
            disp_files = sorted(glob(data_dir + '/disparity-map-half-size/*.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            if mode =="training":
                sample['left'] = left_files[i]
                sample['right'] = right_files[i]
            
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class MS2(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/MS2/',
                 filelist='/data/ywang/my_projects/unimatch/filelists/MS2/',
                 transform=None,
                 mode="training",
                 supervised_type="self_supervised",
                 ):
        super(MS2, self).__init__(transform=transform,  is_MS2=True,
                                            supervised_type=supervised_type)

        assert mode in ['training', 'testing']

        modality = 'rgb'
        if mode != "testing":
            data_list_file = filelist + 'train_day_list.txt'
        
        if mode == "testing":
            data_list_file = filelist + 'val_day_list.txt'

        seq_data_list = [seq_name[:-1] for seq_name in open(data_list_file)] 
        root_disp = data_dir + 'proj_depth'
        root_rgb = data_dir + 'images'

        left_files = []
        right_files = []
        disp_files = []

        for seq in seq_data_list: # iterate over each sequence
            left_files += sorted(glob(os.path.join(root_rgb, seq, modality, 'img_left/*.png')))
            right_files += sorted(glob(os.path.join(root_rgb, seq, modality, 'img_right/*.png')))
            disp_files += sorted(glob(os.path.join(root_disp, seq, modality, 'disp_filtered/*.png')))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            if mode == "training":
                sample['left'] = left_files[i]
                sample['right'] = right_files[i]

            sample['disp'] = disp_files[i]

            self.samples.append(sample)



class RobustMS2(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/MS2/robust_train/',
                 filelist=None,
                 transform=None,
                 mode="training",
                 supervised_type="pseudo_supervised",
                 ):
        super(RobustMS2, self).__init__(transform=transform, is_MS2=True, 
                                            supervised_type=supervised_type)

        assert mode in ['training', 'testing']

        left_files = []
        right_files = []
        disp_files = []

        # # total: 172770
        if mode != 'testing':
            left_files = sorted(glob(data_dir + 'clear/*/img_left/*.png'))
            right_files = sorted(glob(data_dir + 'clear/*/img_right/*.png'))
            disp_files = sorted(glob(data_dir + 'disparity_map/*/disp_filtered/*.png'))

        if mode == 'testing':
            left_files = sorted(glob(data_dir + '/*/img_left/*.png'))
            right_files = sorted(glob(data_dir + '/*/img_right/*.png'))
            disp_files = sorted(glob(data_dir + '/*/disp_filtered/*.png'))
            
        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            if mode == "training":
                sample['left'] = left_files[i]
                sample['right'] = right_files[i]
            
            sample['disp'] = disp_files[i]

            self.samples.append(sample)




class RobustMS2_test(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/MS2/adverse_weather/',
                 filelist=None,
                 transform=None,
                 mode="training",
                 supervised_type="supervised",
                 ):
        super(RobustMS2_test, self).__init__(transform=transform, is_MS2=True, 
                                            supervised_type=supervised_type)

        assert mode in ['training', 'testing']

        left_files = []
        right_files = []
        disp_files = []

        # # total: 172770
        if mode != 'testing':
            left_files = sorted(glob(data_dir + 'night/*/img_left/*.png'))
            right_files = sorted(glob(data_dir + 'night/*//img_right/*.png'))
            disp_files = sorted(glob(data_dir + 'night/*//disp_filtered/*.png'))

        if mode == 'testing':
            left_files = sorted(glob(data_dir + '/*/rgb/img_left/*.png'))
            right_files = sorted(glob(data_dir + '/*/rgb/img_right/*.png'))
            disp_files = sorted(glob(data_dir + '/*/rgb/disp_filtered/*.png'))
            
        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            if mode == "training":
                sample['left'] = left_files[i]
                sample['right'] = right_files[i]
            
            sample['disp'] = disp_files[i]

            self.samples.append(sample)



class SintelStereo(StereoDataset):
    def __init__(self,
                 data_dir='datasets/SintelStereo',
                 split='clean',
                 transform=None,
                 save_filename=False,
                 ):
        super(SintelStereo, self).__init__(transform=transform, is_sintel=True)

        self.save_filename = save_filename

        assert split in ['clean', 'final']

        # total: clean & final each 1064
        left_files = sorted(glob(data_dir + '/training/' + split + '_left/*/*.png'))
        right_files = sorted(glob(data_dir + '/training/' + split + '_right/*/*.png'))
        disp_files = sorted(glob(data_dir + '/training/disparities/*/*.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            if self.save_filename:
                sample['left_name'] = left_files[i]

            self.samples.append(sample)


class ETH3DStereo(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/eth3d',
                 mode='training',
                 transform=None,
                 save_filename=False,
                 ):
        super(ETH3DStereo, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        assert mode in ['training', 'test']
        
        if mode == 'training':
            left_files = sorted(glob(data_dir + '/two_view_training/*/im0.png'))
            right_files = sorted(glob(data_dir + '/two_view_training/*/im1.png'))
        else:
            left_files = sorted(glob(data_dir + '/two_view_test/*/im0.png'))
            right_files = sorted(glob(data_dir + '/two_view_test/*/im1.png'))

        if mode == 'training':
            disp_files = sorted(glob(data_dir + '/two_view_training/*/disp0GT.pfm'))
            assert len(left_files) == len(right_files) == len(disp_files)
        else:
            assert len(left_files) == len(right_files)

        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            
            if mode == 'training':
                sample['disp'] = disp_files[i]

            if save_filename:
                sample['left_name'] = left_files[i]

            self.samples.append(sample)


class MiddleburyEval3(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/Middlebury_half',
                 mode='training',
                 resolution='H',
                 transform=None,
                 save_filename=False,
                 ):
        super(MiddleburyEval3, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        assert mode in ['training', 'test']
        assert resolution in ['Q', 'H', 'F']

        left_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/im0.png'))
        right_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/im1.png'))

        if mode == 'training':
            disp_files = sorted(glob(data_dir + '/' + mode + resolution + '/*/disp0GT.pfm'))
            assert len(left_files) == len(right_files) == len(disp_files)
        else:
            assert len(left_files) == len(right_files)

        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]


            if mode == 'training':
                sample['disp'] = disp_files[i]

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)


class Middlebury20052006(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2005',
                 transform=None,
                 save_filename=False,
                 ):
        super(Middlebury20052006, self).__init__(transform=transform, is_raw_disp_png=True)

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            # Middlebury/2005/Art
            sample = dict()

            sample['left'] = os.path.join(data_dir, curr_dir, 'view1.png')
            sample['right'] = os.path.join(data_dir, curr_dir, 'view5.png')
            sample['disp'] = os.path.join(data_dir, curr_dir, 'disp1.png')

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)

            # same disp for different images
            gt_disp = os.path.join(data_dir, curr_dir, 'disp1.png')

            # also include different illuminations
            for illum in ['Illum1', 'Illum2', 'Illum3']:
                for exp in ['Exp0', 'Exp1', 'Exp2']:
                    # Middlebury/2005/Art/Illum1/Exp0/
                    sample = dict()

                    sample['left'] = os.path.join(data_dir, curr_dir, illum, exp, 'view1.png')
                    sample['right'] = os.path.join(data_dir, curr_dir, illum, exp, 'view5.png')
                    sample['disp'] = gt_disp

                    if save_filename:
                        sample['left_name'] = sample['left']

                    self.samples.append(sample)


class Middlebury2014(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2014',
                 transform=None,
                 save_filename=False,
                 half_resolution=True,
                 ):
        super(Middlebury2014, self).__init__(transform=transform, is_middlebury_eth3d=True,
                                             half_resolution=half_resolution,
                                             )

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            for data_type in ['', 'E', 'L']:
                sample = dict()

                sample['left'] = os.path.join(data_dir, curr_dir, 'im0.png')
                sample['right'] = os.path.join(data_dir, curr_dir, 'im1' + '%s.png' % data_type)
                sample['disp'] = os.path.join(data_dir, curr_dir, 'disp0.pfm')

                if save_filename:
                    sample['left_name'] = sample['left']

                self.samples.append(sample)


class Booster(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/booster_gt/train/balanced/',
                 transform=None,
                 save_filename=False,
                 quater_resolution=True,
                 ):
        super(Booster, self).__init__(transform=transform, is_booster=True,
                                             quater_resolution=quater_resolution,
                                             )


        image1_list = sorted(glob(os.path.join(data_dir, "*/camera_00/im*.png"), recursive=True))
        image2_list = sorted(glob(os.path.join(data_dir, "*/camera_02/im*.png"), recursive=True))

        disp_list = [os.path.join(os.path.split(x)[0], '../disp_00.npy') for x in image1_list]

        assert len(image1_list) == len(image2_list) == len(disp_list) > 0
        
        num_samples = len(image1_list)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = image1_list[i]
            sample['right_ori'] = image2_list[i]

            sample['left'] = image1_list[i]
            sample['right'] = image2_list[i]
            sample['disp'] = disp_list[i]

            self.samples.append(sample)



class Middlebury2021(StereoDataset):
    def __init__(self,
                 data_dir='datasets/Middlebury/2021/data',
                 transform=None,
                 save_filename=False,
                 ):
        super(Middlebury2021, self).__init__(transform=transform, is_middlebury_eth3d=True)

        self.save_filename = save_filename

        dirs = [curr_dir for curr_dir in sorted(os.listdir(data_dir)) if not curr_dir.endswith('.zip')]

        for curr_dir in dirs:
            # Middlebury/2021/artroom1
            sample = dict()

            sample['left'] = os.path.join(data_dir, curr_dir, 'im0.png')
            sample['right'] = os.path.join(data_dir, curr_dir, 'im1.png')
            sample['disp'] = os.path.join(data_dir, curr_dir, 'disp0.pfm')

            if save_filename:
                sample['left_name'] = sample['left']

            self.samples.append(sample)

            # same disp for different images
            gt_disp = os.path.join(data_dir, curr_dir, 'disp0.pfm')

            # Middlebury/2021/data/artroom1/ambient/F0
            curr_img_dir = os.path.join(data_dir, curr_dir, 'ambient')

            # also include different illuminations
            # for data_type in ['F0', 'L0', 'L1', 'L2', 'T0']:
            # only use 'L0' lighting since others are too challenging
            for data_type in ['L0']:
                img0s = sorted(glob(curr_img_dir + '/' + data_type + '/im0e*.png'))

                for img0 in img0s:
                    sample = dict()

                    sample['left'] = img0
                    sample['right'] = img0.replace('/im0', '/im1')
                    assert os.path.isfile(sample['right'])

                    sample['disp'] = gt_disp

                    if save_filename:
                        sample['left_name'] = sample['left']

                    self.samples.append(sample)


class CREStereoDataset(StereoDataset):
    def __init__(self,
                 data_dir='datasets/CREStereo/stereo_trainset/crestereo',
                 transform=None,
                 ):
        super(CREStereoDataset, self).__init__(transform=transform, is_crestereo=True)

        left_files = sorted(glob(data_dir + '/*/*_left.jpg'))
        right_files = sorted(glob(data_dir + '/*/*_right.jpg'))
        disp_files = sorted(glob(data_dir + '/*/*_left.disp.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class TartanAir(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/Tartanair',
                 transform=None,
                 ):
        super(TartanAir, self).__init__(transform=transform, is_tartanair=True)

        left_files = sorted(glob(data_dir + '/*/*/*/*/image_left/*.png'))
        right_files = sorted(glob(data_dir + '/*/*/*/*/image_right/*.png'))
        disp_files = sorted(glob(data_dir + '/*/*/*/*/depth_left/*.npy'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class CARLA(StereoDataset):
    def __init__(self,
                 data_dir='/data/ywang/dataset/HR-VS-Stereo/carla-highres/trainingF',
                 transform=None,
                 ):
        super(CARLA, self).__init__(transform=transform, is_middlebury_eth3d=True,
                                    quater_resolution=True)

        left_files = sorted(glob(data_dir + '/*/im0.png'))
        right_files = sorted(glob(data_dir + '/*/im1.png'))
        disp_files = sorted(glob(data_dir + '/*/disp0GT.pfm'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left_ori'] = left_files[i]
            sample['right_ori'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class InStereo2K(StereoDataset):
    def __init__(self,
                 data_dir='datasets/InStereo2K',
                 transform=None,
                 ):
        super(InStereo2K, self).__init__(transform=transform, is_instereo2k=True)

        # merge train and test
        left_files = sorted(glob(data_dir + '/train/*/*/left.png') + glob(data_dir + '/test/*/left.png'))
        right_files = sorted(glob(data_dir + '/train/*/*/right.png') + glob(data_dir + '/test/*/right.png'))
        disp_files = sorted(glob(data_dir + '/train/*/*/left_disp.png') + glob(data_dir + '/test/*/left_disp.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


class FallingThings(StereoDataset):
    def __init__(self,
                 data_dir='datasets/FallingThings',
                 transform=None,
                 ):
        super(FallingThings, self).__init__(transform=transform, is_fallingthings=True)

        # merge train and test
        left_files = sorted(glob(data_dir + '/*/*/*left.jpg'))
        right_files = sorted(glob(data_dir + '/*/*/*right.jpg'))
        disp_files = sorted(glob(data_dir + '/*/*/*left.depth.png'))

        assert len(left_files) == len(right_files) == len(disp_files)
        num_samples = len(left_files)

        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['disp'] = disp_files[i]

            self.samples.append(sample)


def build_dataset(args):
    if args.stage == 'sceneflow':
        train_transform_list =  [ transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                # transforms.RandomMaskOcclusion(),
                                transforms.RandomOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        things = FlyingThings3D(transform=train_transform)
        monkaa = Monkaa(transform=train_transform)
        driving = Driving(transform=train_transform)

        train_dataset = 10 * things + 10 * monkaa + 10 * driving

        return train_dataset

    elif args.stage == 'vkitti2':
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.RandomOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        train_dataset = VKITTI2(transform=train_transform)
        train_dataset += Driving(transform=train_transform)
        train_dataset += 10 * CARLA(transform=train_transform)
        # train_dataset +=  DrivingStereo(transform=train_transform)

        return train_dataset
    
    elif args.stage == 'drivingstereo':
        train_transform_list = [ transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.RandomOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]
        
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = DrivingStereo(transform=train_transform, supervised_type=args.supervised_type)
        return train_dataset


    elif args.stage == 'robust_drivingstereo':
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.RandomOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]
        
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = RobustDrivingStereo(transform=train_transform, supervised_type=args.supervised_type)
        train_dataset +=  2 * VKITTI2(transform=train_transform)
        # train_dataset += 20 * CARLA(transform=train_transform)
        return train_dataset



    elif args.stage == 'MS2':
        train_transform_list = [ transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]
        
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = MS2(transform=train_transform, supervised_type=args.supervised_type)
        train_dataset +=  2 * VKITTI2(transform=train_transform)
        
        return train_dataset


    elif args.stage == 'robust_MS2':
        train_transform_list = [ transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]
        
        train_transform = transforms.Compose(train_transform_list)
        train_dataset = RobustMS2(transform=train_transform, supervised_type=args.supervised_type)
        train_dataset +=  2 * VKITTI2(transform=train_transform)
        return train_dataset


    elif args.stage == 'kitti15mix':
        train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomMaskOcclusion(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]

        train_transform = transforms.Compose(train_transform_list)

        # kitti15: 200
        kitti15 = KITTI15(transform=train_transform,
                          )
        # kitti12: 195
        kitti12 = KITTI12(transform=train_transform,
                          )
        vkitti2 = VKITTI2(transform=train_transform)
        train_dataset = 200 * kitti15 + 200 * kitti12 + 1 * vkitti2

        return train_dataset


    else:
        raise NotImplementedError
