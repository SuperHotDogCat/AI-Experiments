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
    reshaped = pose_batch.reshape(-1, 64, 2, shape[-1]/2)
    diff = reshaped[:,1:] - reshaped[:,:-1] #動きの差分
    return diff.reshape(-1, 63, shape[-1])

def keypoints_to_train(poses, arr):
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
    repeats[1] = 2 # dim = 1のところを2回繰り返す
    return input.repeat(*repeats)

def ResidualConnectUpSampling1D(nn.Module):
    def __init__(self, in_channels, out_channels, type='1d', leaky=False, downsample=False, norm='batch', k=None, \
                 s=None, padding='same', G = None):
        super().__init__()
        self.convnormrelu = ConvNormRelu(in_channels, out_channels, type=type, leaky=leaky, downsample=downsample, norm=norm, k=k, s=s, padding=padding, G=G)
    def forward(self, input_data):
        input_data = UpSampling1D(input_data) + input_data
        input_data = self.convnormrelu(input_data)
        return input_data