import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torchvision import transforms
import torchvision
from tqdm import tqdm
class GNet(nn.Module):
    def __init__(self, z_size, num_filters) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_size, num_filters*4, 4,1,0,bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(num_filters*4,num_filters*2,3,2,1,bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(num_filters*2,num_filters,4,2,1,bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(num_filters,3,4,2,1,bias=False),
            nn.Tanh()
        )
    def forward(self,fake_data):
        """
        create_noise関数は(batch_size, z_size)のため、(batch_size, z_size, 1, 1)に入力を変更してから計算を行う
        """
        batch_size = fake_data.size(0)
        z_size = fake_data.size(1)
        fake_data = fake_data.view(batch_size,z_size,1,1)
        fake_img = self.model(fake_data)
        return fake_img.squeeze(1)

class DNet(nn.Module):
    def __init__(self,num_filters, image_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,num_filters,4,2,1,bias=False),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters,num_filters*2,4,2,1,bias=False),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters*2,num_filters*4,3,2,1,bias=False),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters*4,3,4,1,0,bias=False),
            nn.Sigmoid(),
        )
        self.image_size = image_size
    def forward(self, data):
        data = data.view(-1,1,*self.image_size)
        out = self.model(data)
        return out.view(-1,1)