import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from audio_to_multiple_pose_gan.torch_layers import ConvNormRelu, UpSampling1D, UNet1D, UNet1DGAN

from einops import rearrange
import torchaudio
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
def torch_mel_spectograms(X_audio):
    if not isinstance(X_audio, torch.Tensor):
        X_audio = torch.tensor(X_audio)
    n_fft = 512
    win_length = 400
    hop_length = 160
    stft = torch.stft(X_audio, window=torch.hann_window(window_length=win_length).to(device), win_length = win_length, n_fft=n_fft, return_complex=True, hop_length=hop_length, center=False).to(device)
    stft = torch.abs(stft)
    stft = rearrange(stft, "a b c->a c b")
    sr = 16000
    mel_spect_input = torchaudio.functional.melscale_fbanks(n_freqs=stft.size(2), n_mels=64, f_min=125., f_max=7500., sample_rate=sr).to(device)
    input_data = stft @ mel_spect_input
    input_data = torch.log(input_data + 1e-6)
    input_data = input_data.unsqueeze(-1)
    input_data = rearrange(input_data, "b h w c->b c h w")
    return input_data
"""
input_dataはTensorflow準拠で作ったため、場合によってはTensorの形状を変更するべきなことに注意
例: Tensorflow: (batch, height, width, channels), Pytorch: (batch, channels, height, width)
"""
class D_patchgan(nn.Module):
    def __init__(self, in_channels, n_downsampling=2, norm='batch', reuse=False, is_training=False,linear_size = 24):
        super().__init__()
        ndf = 64
        self.conv1d = nn.Conv1d(in_channels, out_channels=ndf, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        modulelist = nn.ModuleList([])
        for n in range(1, n_downsampling):
            nf_mult = min(2**n, 8)
            if n == 1:
                modulelist.append(ConvNormRelu(ndf, ndf * nf_mult, type = "1d", downsample=True, norm=norm, leaky=True, padding=1))
                prev_channels = ndf * nf_mult
            else:
                modulelist.append(ConvNormRelu(prev_channels, ndf * nf_mult, type = "1d", downsample=True, norm=norm, leaky=True, padding = 1))
                prev_channels = ndf * nf_mult
        nf_mult = min(2**n_downsampling, 8)
        modulelist.append(ConvNormRelu(prev_channels, ndf * nf_mult, type = "1d", norm=norm, leaky=True, k=4, s=1))
        modulelist.append(nn.Conv1d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding="same"))
        self.convnormrelu = nn.Sequential(*modulelist)
        #元コードに流石に無理があった気がしたのでsigmoidを追加する
        self.linear = nn.Linear(linear_size,1) #ハードコーディング, linear_size = 24はnewkeypointsなし, 
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_data):
        #このD_patchganはDiscriminator関数である。
        #入力は(batch, time, features)で問題なさそう。特にposeを変える必要はなし
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data).float()
        input_data = self.conv1d(input_data)
        input_data = self.leaky_relu(input_data)
        input_data = self.convnormrelu(input_data)
        input_data = rearrange(input_data, "b l f->b (l f)")
        input_data = self.linear(input_data)
        input_data = self.sigmoid(input_data)
        return input_data #真のデータかいなかを判別する確率になっている

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

        #ここが違う
        self.downsampling_block5 = UNet1D(in_channels=256, out_channels=256)

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
            ConvNormRelu(in_channels=256, out_channels=256, ),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, x_audio,):
        #x_audio = input_dict['audio']
        input_data = torch_mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.downsampling_block5(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)
        input_data = rearrange(input_data, "b s o->b o s")
        return input_data
    
class Audio2PoseGANS(nn.Module):
    def __init__(self, in_channels, out_channels = 98, reuse = False, is_Training = False, norm = 'batch', pose_size = 64):
        super().__init__()
        #paddingがpytorchだと保護されていないようなので治すこと
        self.downsampling_block1 = nn.Sequential(
            ConvNormRelu(in_channels=in_channels, out_channels=64, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=64, out_channels=64, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block2 = nn.Sequential(
            ConvNormRelu(in_channels=64, out_channels=128, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=128, out_channels=128, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block3 = nn.Sequential(
            ConvNormRelu(in_channels=128, out_channels=256, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
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

        self.downsampling_block5 = UNet1DGAN(in_channels=256, out_channels=256, leaky=True, norm=norm)

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, x_audio,):
        #x_audio = input_dict['audio']
        input_data = torch_mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.downsampling_block5(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)
        input_data = rearrange(input_data, "b s o->b o s")
        return input_data

class Audio2PoseGANSTransformer(nn.Module):
    def __init__(self, in_channels, out_channels = 98, reuse = False, is_Training = False, norm = 'batch', pose_size = 64):
        super().__init__()
        #paddingがpytorchだと保護されていないようなので治すこと
        self.downsampling_block1 = nn.Sequential(
            ConvNormRelu(in_channels=in_channels, out_channels=64, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=64, out_channels=64, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block2 = nn.Sequential(
            ConvNormRelu(in_channels=64, out_channels=128, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=128, out_channels=128, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block3 = nn.Sequential(
            ConvNormRelu(in_channels=128, out_channels=256, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
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

        encoderlayer = nn.TransformerEncoderLayer(64, nhead=4, dim_feedforward=64 * 4, activation="gelu",batch_first=True, bias=False)

        self.transformerencoder = nn.TransformerEncoder(encoderlayer, num_layers=2,)

        self.downsampling_block5 = UNet1DGAN(in_channels=256, out_channels=256, leaky=True, norm=norm)

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

    def forward(self, x_audio,):
        #x_audio = input_dict['audio']
        input_data = torch_mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.transformerencoder(input_data)
        input_data = self.downsampling_block5(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)
        input_data = rearrange(input_data, "b s o->b o s")
        return input_data
    
class Audio2PoseGANS_STTransformer(nn.Module):
    def __init__(self, in_channels, out_channels = 98, reuse = False, is_Training = False, norm = 'batch', pose_size = 64, pose_embed_size = 104, nhead = 4):
        super().__init__()
        #paddingがpytorchだと保護されていないようなので治すこと
        self.downsampling_block1 = nn.Sequential(
            ConvNormRelu(in_channels=in_channels, out_channels=64, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=64, out_channels=64, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block2 = nn.Sequential(
            ConvNormRelu(in_channels=64, out_channels=128, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=128, out_channels=128, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
        )
        self.downsampling_block3 = nn.Sequential(
            ConvNormRelu(in_channels=128, out_channels=256, type='2d', norm=norm, leaky=True, downsample=False),
            ConvNormRelu(in_channels=256, out_channels=256, type='2d', norm=norm, leaky=True, downsample=True, padding=1),
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

        encoderlayer = nn.TransformerEncoderLayer(64, nhead=nhead, dim_feedforward=64 * 4, activation="gelu",batch_first=True, bias=False)
        self.transformerencoder = nn.TransformerEncoder(encoderlayer, num_layers=4,)

        self.linear = nn.Linear(64, pose_embed_size)

        decoderlayer = nn.TransformerDecoderLayer(pose_embed_size, nhead=nhead, dim_feedforward=pose_embed_size * 4, activation="gelu",batch_first=True, bias=False)
        self.transformerdecoder = nn.TransformerDecoder(decoderlayer, num_layers=4,)

        self.positioning_encoding_src = nn.Parameter(torch.zeros(1, 64, pose_embed_size))
        self.positioning_encoding_tgt = nn.Parameter(torch.zeros(1, 65, pose_embed_size))

        self.decoder = nn.Sequential(
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
            ConvNormRelu(in_channels=256, out_channels=256, leaky=True, downsample=False, norm=norm),
        )

        self.logits = nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding="same")
    def forward(self, x_audio,Y_pose, device = device, mask = True):
        #x_audio = input_dict['audio']
        """
        	位置エンコーディングが必要かもしれない
        """
        input_data = torch_mel_spectograms(x_audio) #必ず何かしらを書く
        input_data = self.downsampling_block1(input_data)
        input_data = self.downsampling_block2(input_data)
        input_data = self.downsampling_block3(input_data)
        input_data = self.downsampling_block4(input_data)
        input_data = self.resize(input_data)
        input_data = torch.squeeze(input_data, dim = 3)
        input_data = self.transformerencoder(input_data)
        input_data = self.decoder(input_data)
        input_data = self.logits(input_data)
        input_data = rearrange(input_data, "b s o->b o s")

        input_data = input_data + self.positioning_encoding_src[:,:input_data.size(1),:]
        if mask:
            mask = self.create_mask(Y_pose, device = device)
            Y_pose = Y_pose + self.positioning_encoding_tgt[:,:Y_pose.size(1),:]
            input_data = self.transformerdecoder(tgt = Y_pose, memory = input_data, tgt_is_causal = True, tgt_mask = mask)
        else:
            Y_pose = Y_pose + self.positioning_encoding_tgt[:,:Y_pose.size(1),:]
            input_data = self.transformerdecoder(tgt = Y_pose, memory = input_data)

        return input_data
    
    def create_mask(self, x: torch.tensor, device: str = device):
        """
        (batch_size, sequence_length, embedding_dim)の入力を想定
        """
        """
        Trueが無視される値であることに注意すること
        """
        seq_len = x.size(1)
        #srcのマスク制作
        mask = torch.triu(torch.ones(size = (seq_len, seq_len))==1).transpose(0,1) #下三角行列を作る
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask==1.,float(0.0)).to(device)
        return mask
    
    @torch.no_grad
    def predict(self, x_audio: torch.Tensor, pose_size = 104, seq_len = 64, device = device, begin_pose: np.ndarray = None):
        self = self.train()
        self.downsampling_block1.eval()
        self.downsampling_block2.eval()
        self.downsampling_block3.eval()
        self.downsampling_block4.eval()
        self.resize.eval()
        self.transformerdecoder.eval()
        self.decoder.eval()
        self.logits.eval()

        if len(x_audio.shape) != 2:
            x_audio = x_audio.unsqueeze(0)
        try:
            output = torch.tensor(begin_pose).to(device)
        except:
            output = torch.zeros(size=(x_audio.size(0), 1, pose_size)).to(device)
        for i in range(seq_len):
            pred_pose = self.forward(x_audio, output,device=device, mask=False)
            output = torch.concatenate([output, pred_pose[:,[-1],:]], dim = 1)
        self = self.train()
        return output[:,1:,:]