import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
import timm
import math

  

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x, dual_inp=False):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x

class MultiBasicEncoder(nn.Module):
    def __init__(self, input_dim=[128], output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        
        output_list = []
    
        self.vfm_04 =None
        self.vfm_04 = nn.Sequential(nn.Conv2d(input_dim[3] + 128, output_dim[0][3], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][3]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs04 = nn.ModuleList(output_list)

        self.vfm_08 =None
        self.vfm_08 = nn.Sequential(nn.Conv2d(input_dim[2] + 128, output_dim[0][2], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][2]),
                                        nn.ReLU(inplace=True)
                                        )
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        self.vfm_16 =None
        self.vfm_16 = nn.Sequential(nn.Conv2d(input_dim[1] + 128, output_dim[0][1], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][1]),
                                        nn.ReLU(inplace=True)
                                        )
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, vfm_features, x):
        b, _, _, _ = x.shape
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        
        x_2res = self.layer2(x)
        x_4res = self.layer3(x_2res)
        x_8res = self.layer4(x_4res)
        x_16res = self.layer5(x_8res)

        if self.vfm_04:
            vfm_outputs4 = vfm_features[3]
            vfm_outputs_fused4 = self.vfm_04(torch.cat([x_4res, vfm_outputs4], dim=1))  
            outputs04 = [f(vfm_outputs_fused4) for f in self.outputs04]
            
        if self.vfm_08:
            vfm_outputs8 = vfm_features[2]
            vfm_outputs_fused8 = self.vfm_08(torch.cat([x_8res, vfm_outputs8], dim=1))
            outputs08 = [f(vfm_outputs_fused8) for f in self.outputs08]

        if self.vfm_16:
            vfm_outputs16 = vfm_features[1]
            vfm_outputs_fused16 = self.vfm_16(torch.cat([x_16res, vfm_outputs16], dim=1))
            outputs16 = [f(vfm_outputs_fused16) for f in self.outputs16]

        return (outputs04, outputs08, outputs16)

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MatchingDecoder(nn.Module):
    def __init__(self, model_size='base', input_dim=128, output_dim=[128], norm_fn='instance', dropout=0.0, downsample=3):
        super(MatchingDecoder, self).__init__()
        self.model_size = model_size
        self.norm_fn = norm_fn
        self.downsample = downsample

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        
        self.layer_1 = self._make_layer(64, stride=1)
        self.layer_2 = self._make_layer(96, stride=2)
        self.layer_3 = self._make_layer(128, stride=1)
        
        self.final_output = nn.Sequential(
            nn.Conv2d(128 + input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, vfm_output, x, dual_inp=False):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        
        vfm_output = torch.cat([vfm_output, x], dim=1)
        output = self.final_output(vfm_output)
        
        if is_list:
            output = output.split(split_size=batch_dim, dim=0)

        return output




class MultiVFMDecoder(nn.Module):
    def __init__(self, input_dim=[128], output_dim=[128], norm_fn='batch', dropout=0.0):
        super(MultiVFMDecoder, self).__init__()
        self.norm_fn = norm_fn

        output_list = []
        
        self.vfm_08 =None
        if input_dim[2] != output_dim[0][2]:
            self.vfm_08 = nn.Sequential(nn.Conv2d(input_dim[2], output_dim[0][2], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][2]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(dim[2], dim[2], self.norm_fn, stride=1),
                nn.Conv2d(dim[2], dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        
        self.vfm_16 = None
        if input_dim[1] != output_dim[0][1]:
            self.vfm_16 = nn.Sequential(nn.Conv2d(input_dim[1], output_dim[0][1], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][1]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(dim[1], dim[1], self.norm_fn, stride=1),
                nn.Conv2d(dim[1], dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        
        self.vfm_32 = None
        if input_dim[0] != output_dim[0][0]:
            self.vfm_32 = nn.Sequential(nn.Conv2d(input_dim[0], output_dim[0][0], 3, padding=1),
                                        nn.InstanceNorm2d(output_dim[0][0]),
                                        nn.ReLU(inplace=True)
                                        )
            
        for dim in output_dim:
            conv_out = nn.Conv2d(dim[0], dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, vfm_output, num_layers=3):
        vfm_outputs08 = vfm_output[-1]
        if self.vfm_08:
            vfm_outputs08 = self.vfm_08(vfm_outputs08)
        outputs08 = [f(vfm_outputs08) for f in self.outputs08]
        if num_layers == 1:
            return outputs08

        vfm_outputs16 = vfm_output[-2]
        if self.vfm_16:
            vfm_outputs16 = self.vfm_08(vfm_outputs16)
        outputs16 = [f(vfm_outputs16) for f in self.outputs16]

        if num_layers == 2:
            return outputs08, outputs16

        vfm_outputs32 = vfm_output[-3]
        if self.vfm_32:
            vfm_outputs32 = self.vfm_32(vfm_outputs32)
        outputs32 = [f(vfm_outputs32) for f in self.outputs32]

        return outputs08, outputs16, outputs32
