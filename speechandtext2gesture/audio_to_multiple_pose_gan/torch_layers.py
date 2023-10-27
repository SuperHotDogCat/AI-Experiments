import torch
import torch.nn as nn
import torch.nn.functional as F
"""
https://github.com/amirbar/speech2gestureのaudio_to_multiple_pose_ganのtorch版を書く
"""
"""
元リポジトリは全て関数で実装するTensorflow関数API型の実装を行なっていたが、今回はPytorchのため、
Pytorchらしくclassでの実装を行う
"""
def to_motion_delta(pose_batch):
    """
    poseを変換する関数 <-おそらく動きの差分に
    """
    shape = pose_batch.shape
    reshaped = pose_batch.reshape(-1, 64, 2, shape[-1]//2)
    diff = reshaped[:,1:] - reshaped[:,:-1] #動きの差分
    return diff.reshape(-1, 63, shape[-1])

def to_motion_delta_2(pose_batch):
    """
    poseを変換する関数 <-おそらく動きの差分に
    train transformer用
    """
    shape = pose_batch.shape
    reshaped = pose_batch.reshape(-1, 63, 2, shape[-1]//2)
    diff = reshaped[:,1:] - reshaped[:,:-1] #動きの差分
    return diff.reshape(-1, 62, shape[-1])

def keypoints_to_train(poses, arr):
    """
    arrにはmodel.pyで実装された_get_training_keypointsのリストが入る
    """
    shape = poses.shape
    reshaped = poses.reshape(shape[0], shape[1], 2, 49)
    required_keypoints = torch.gather(reshaped, dim = 3, index = arr)
    return required_keypoints.reshape(shape[0], shape[1], 2*len(arr))

def keypoints_regloss(gt_keypoints, pred_keypoints, regloss_type):
    """
    損失関数を定義する
    gt_keypoints: ground truth
    pred_keypoints
    """
    residual = torch.flatten(gt_keypoints) - torch.flatten(pred_keypoints)
    if regloss_type == "l1":
        return torch.abs(residual).mean()
    elif regloss_type == "l2":
        return torch.pow(residual, 2).mean()
    else:
        raise ValueError("Wrong regression loss")



class Norm(nn.Module):
    def __init__(self, channels, norm='batch', type='1d', G = None):
        super().__init__()
        if norm == 'batch':
            if type == '2d':
                self.norm = nn.BatchNorm2d(channels) #input shape: (N,C,H,W) = (batch_size, channels, height, width)
            elif type == '1d':
                self.norm = nn.BatchNorm1d(channels) #input shape: (N,C,L) = (batch_size, channels(features), sequence_length)
            else:
                raise ValueError('Unimplemented Norm type.')
        elif norm == 'instance':

            #元のリポジトリではinstance normは訓練不可能(trainable == False)にしていたが........

            if type == '2d':
                self.norm = nn.InstanceNorm2d(channels) 
            elif type == '1d':
                self.norm = nn.InstanceNorm1d(channels)
            else:
                raise ValueError('Unimplemented Norm type.')
        elif norm == 'group':
            if G == None:
                raise ValueError('if norm == group, G must not be None.')
            #元のリポジトリではdefaultのGは32だった
            self.norm = nn.GroupNorm(G, channels)
        else:
            raise ValueError('Unimplemented Norm type.')
    def forward(self, input):
        return self.norm(input)

class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, type='1d', leaky=False, downsample=False, norm='batch', k=None,\
                 s=None, padding='same', G = None):
        super().__init__()
        if k == None and s == None:
            if not downsample:
                k = 3
                s = 1
            else:
                k = 4
                s = 2

        #元々convはkernelがglorot, biasがzeroで初期化だったがとりあえず無視する
        if type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding,) #input shape: (N,C,L) = (batch_size, in_channels(features), sequence_length)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding,) #input shape: (N,C,H,W) = (batch_size, in_channels, height, width)
        else:
            raise ValueError('Unimplemented conv type.')
        self.norm = Norm(out_channels, norm = norm, type = type, G = G)

        if leaky:
            self.relu = nn.LeakyReLU(0.2)
        else:
            self.relu = nn.ReLU()
    def forward(self, input):
        input = self.conv(input)
        input = self.norm(input)
        input = self.relu(input)
        return input

def UpSampling1D(input):
    repeats = [1 for _ in range(len(input.shape))] #(1,1,1,1,....)
    repeats[2] = 2 # dim = 2のところを2回繰り返す
    return input.repeat(*repeats)

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=False, norm='batch', k=None, \
                 s=None, padding='same', G = None):
        super().__init__()
        self.convnormrelus = nn.ModuleList([ConvNormRelu(in_channels, out_channels, type="1d", leaky=leaky,downsample=False, norm=norm, k=k, s=s, padding=padding, G=G) for _ in range(12)])
        self.maxpool1ds = nn.ModuleList([nn.MaxPool1d(2,2) for _ in range(5)])
    def forward(self, x):
        x1 = self.convnormrelus[0](x)
        x1 = self.convnormrelus[1](x1)

        x2 = self.maxpool1ds[0](x1)
        x2 = self.convnormrelus[2](x2)

        x3 = self.maxpool1ds[1](x2)
        x3 = self.convnormrelus[3](x3)

        x4 = self.maxpool1ds[2](x3)
        x4 = self.convnormrelus[4](x4)

        x5 = self.maxpool1ds[3](x4)
        x5 = self.convnormrelus[5](x5)

        x6 = self.maxpool1ds[4](x5)
        x6 = self.convnormrelus[6](x6)

        x5 = UpSampling1D(x6) + x5
        x5 = self.convnormrelus[7](x5)

        x4 = UpSampling1D(x5) + x4
        x4 = self.convnormrelus[8](x4)

        x3 = UpSampling1D(x4) + x3
        x3 =self.convnormrelus[9](x3)

        x2 = UpSampling1D(x3) + x2
        x2 = self.convnormrelus[10](x2)

        x1 = UpSampling1D(x2) + x1
        x1 = self.convnormrelus[11](x1)
        return x1

class UNet1DGAN(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=False, norm='batch', k=None, \
                 s=None, padding='same', G = None):
        super().__init__()
        self.convnormrelus_downsamples = nn.ModuleList([ConvNormRelu(in_channels, out_channels, type="1d", leaky=leaky,downsample=True, norm=norm, k=k, s=s, padding=1, G=G) for _ in range(5)])
        self.convnormrelus_nondownsamples = nn.ModuleList([ConvNormRelu(in_channels, out_channels, type="1d", leaky=leaky,downsample=False, norm=norm, k=k, s=s, padding=padding, G=G) for _ in range(7)])
    
    def forward(self, x):
        x1 = self.convnormrelus_nondownsamples[0](x)
        x1 = self.convnormrelus_nondownsamples[1](x1)

        x2 = self.convnormrelus_downsamples[0](x1)

        x3 = self.convnormrelus_downsamples[1](x2)

        x4 = self.convnormrelus_downsamples[2](x3)

        x5 = self.convnormrelus_downsamples[3](x4)

        x6 = self.convnormrelus_downsamples[4](x5)

        x5 = UpSampling1D(x6) + x5
        x5 = self.convnormrelus_nondownsamples[2](x5)

        x4 = UpSampling1D(x5) + x4
        x4 = self.convnormrelus_nondownsamples[3](x4)

        x3 = UpSampling1D(x4) + x3
        x3 =self.convnormrelus_nondownsamples[4](x3)

        x2 = UpSampling1D(x3) + x2
        x2 = self.convnormrelus_nondownsamples[5](x2)

        x1 = UpSampling1D(x2) + x1
        x1 = self.convnormrelus_nondownsamples[6](x1)
        return x1
