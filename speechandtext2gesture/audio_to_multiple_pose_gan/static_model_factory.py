import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from audio_to_multiple_pose_gan.torch_layers import ConvNormRelu, UpSampling1D, ResidualConnectUpSampling1D

class Audio2Pose(nn.Module):
    def __init__(self, in_channels, out_channels = 98, reuse = False, is_Training = False, pose_size = 64):
        super().__init__()
        self.downsampling_block1 = nn.Sequential(
            ConvNormRelu(in_channels=in_channels, out_channels=64, type='2d', ),
            ConvNormRelu(in_channels=64, out_channels=64, type='2d', ),
            nn.MaxPool2d(kernel_size = 2, stride=2),
        )
        self.downsampling_block2 = nn.Sequential(
            ConvNormRelu(in_channels=64, out_channels=128, type='2d', ),
            ConvNormRelu(in_channels=128, out_channels=128, type='2d', ),
            nn.MaxPool2d(kernel_size = 2, stride= 2),
        )
        self.downsampling_block3 = nn.Sequential(
            ConvNormRelu(in_channels=128, out_channels=256, type='2d', ),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', ),
            nn.MaxPool2d(kernel_size = 2, stride= 2),
        )
        self.downsampling_block4 = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', ),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', k = (3,8), s = 1, padding='valid'),
        )
        #To do: downsampling_block4実装のためには元リポジトリのinput_dictを調査する必要がある
        #input_dictを調査するまでこれ以上はかけない
        """
        pose_data: (batch_data, 64, 98)なので、64フレームであると考えられる。
        downsampling_block4以後で(batch_size, channels, height, width)を(batch_size, channels, 64, 1)に変える。
        """
        self.resize = transforms.Resize((pose_size,1), InterpolationMode.BILINEAR)

        self.downsampling_block5 = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            nn.MaxPool1d(kernel_size = 2, stride= 2),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            nn.MaxPool1d(kernel_size = 2, stride= 2),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            nn.MaxPool1d(kernel_size = 2, stride= 2),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            nn.MaxPool1d(kernel_size = 2, stride= 2),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            nn.MaxPool1d(kernel_size = 2, stride= 2),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256),
        )

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, input_dict,):
        x_audio = input_dict['audio']
        input_data = mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.downsampling_block5(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)

        return input_data
    
class Audio2PoseGANS(nn.Module):
    def __init__(self, in_channels, out_channels = 98, reuse = False, is_Training = False, norm = 'batch', pose_size = 64):
        super().__init__()
        self.downsampling_block1 = nn.Sequential(
            ConvNormRelu(in_channels=in_channels, out_channels=64, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=64, out_channels=64, type='2d', norm=norm, leaky=True, downsample=True),
        )
        self.downsampling_block2 = nn.Sequential(
            ConvNormRelu(in_channels=64, out_channels=128, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=128, out_channels=128, type='2d', norm=norm, leaky=True, downsample=True),
        )
        self.downsampling_block3 = nn.Sequential(
            ConvNormRelu(in_channels=128, out_channels=256, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', norm=norm, leaky=True, downsample=True),
        )
        self.downsampling_block4 = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', k = (3,8), s = 1, padding='valid', norm=norm, leaky=True, downsample=False),
        )
        #To do: downsampling_block4実装のためには元リポジトリのinput_dictを調査する必要がある
        #input_dictを調査するまでこれ以上はかけない
        """
        pose_data: (batch_data, 64, 98)なので、64フレームであると考えられる。
        downsampling_block4以後で(batch_size, channels, height, width)を(batch_size, channels, 64, 1)に変える。
        """
        self.resize = transforms.Resize((pose_size,1), InterpolationMode.BILINEAR)

        self.downsampling_block5 = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=True, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=True, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=True, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=True, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=True, norm=norm),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256,leaky=True, downsample=False, norm=norm),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256,leaky=True, downsample=False, norm=norm),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256,leaky=True, downsample=False, norm=norm),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256,leaky=True, downsample=False, norm=norm),
            ResidualConnectUpSampling1D(in_channels = 256, out_channels = 256,leaky=True, downsample=False, norm=norm),
        )

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, input_dict,):
        x_audio = input_dict['audio']
        input_data = mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.downsampling_block5(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)
        
        return input_data