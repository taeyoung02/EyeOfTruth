import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('C:\\Users\\owner\\PycharmProjects\\pythonProject1')
from DJPEG.extract_dct import DCT_basis_torch

class DoubleJpeg(nn.Module):
    def __init__(self):
        super(DoubleJpeg, self).__init__()
        self.dct_basis = DCT_basis_torch()

        # define convolution kernel
        # feature.shape = 120x64x1
        self.conv1a = nn.Conv2d(1, 64, 5, padding=2) # 입력 1, 출력 64, 필터 5x5
        self.conv1b = nn.Conv2d(64, 64, 5, padding=2)
        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        # 60x32x64
        self.conv2a = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2a = nn.BatchNorm2d(128)

        # 30x16x128
        self.conv3a = nn.Conv2d(128, 256,5, padding=2)
        self.bn3a = nn.BatchNorm2d(256)

        # 아핀연산 y = Wx + b
        # 15x8x256+64
        self.fc1 = nn.Linear(15*8*256+64, 500)
        self.fc2 = nn.Linear(564, 500) # bias =64
        self.fc3 = nn.Linear(564, 2)

    def forward(self, x, qvectors): # img, quantization table
        # feature extraction
        with torch.no_grad():
            # x.shape = (32,1,256,256)
            # dct_basis = (64,1,8,8)
            # batch,channel,width,height
            x = F.conv2d(x, weight=self.dct_basis, stride=8) # use dct_basis as filter = get the img's dct coefficients
            # x.shape= (32,64,32,32)

            gamma = 1e+06
            for b in range(-60, 61):
                # set channel to height, accumulate by 121
                X = torch.sum(torch.sigmoid(gamma*(x-b)), axis=[2,3])/1024
                # X.shape = (32,64)
                X = torch.unsqueeze(X, axis=1)
                # X.shape = (32,1,64)
                if b != -60:
                    features = torch.cat([features, X], axis=1)
                    # features.shape = (32,~121,64)
                else:
                    features = X

            features = features[:, 0:120, :] - features[:, 1:121, :]
            features = torch.reshape(features, (-1, 1, 120, 64))

        # conv layers

        x = F.relu(self.bn1a(self.conv1a(features))) # convolution, batch_normalize
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, (2,2)) # maxpool2D

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.max_pool2d(x, (2,2))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.max_pool2d(x, (2,2))
        print(x.shape)
        x_flat = torch.reshape(x, (-1, 15*8*256))
        print(x_flat.shape)
        # fully connected layers

        x = torch.cat([qvectors, x_flat], axis=1) # insert q_tables

        x = F.relu(self.fc1(x))

        x = torch.cat([qvectors, x], axis=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([qvectors, x], axis=1)
        x = self.fc3(x)
        return x


    

