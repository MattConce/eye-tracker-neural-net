# Pytorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import math

class EyeModel(nn.Module):
    def __init__(self):
        super(EyeModel, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.grup = nn.GroupNorm(3, 24)

        self.se1 = SELayer(48, 16)
        self.se2 = SELayer(128, 16)

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0) 
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=5)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.agn1 = AdaGN(128, 48)
        self.agn2 = AdaGN(128, 64)
        self.agn3 = AdaGN(128, 128)
        self.agn4 = AdaGN(128, 64)

    def forward(self, x, ada_in):

        x1 = self.conv1(x)
        x1 = self.grup(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.agn1(x1, 6, ada_in)
        x1 = self.pool(x1)
        x1 = self.se1(x1)
        x1 = self.conv3(x1)
        x1 = self.agn2(x1, 8, ada_in)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x2 = self.conv4(x1)
        x2 = self.agn3(x2, 16, ada_in)
        x2 = self.relu(x2)
        x2 = self.se2(x2)
        x2 = self.conv5(x2)
        x2 = self.agn4(x2, 8, ada_in)
        x2 = self.relu(x2)

        return torch.cat((x1, x2), 1)

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(6, 48),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(12, 96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 192),
            nn.ReLU(),
            SELayer(192, 16),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            SELayer(64, 16),
        )

        self.linear1 = nn.Linear(5 * 5 * 64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_block(x).view(x.size(0), -1)
        x = self.linear1(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        return self.lrelu(x)

class SELayer(nn.Module):
    def __init__(self, ch, ratio):
        super(SELayer, self).__init__()
        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Fully Conected layer L(.)
        self.FC = nn.Sequential(
            nn.Linear(ch, ch // ratio, bias=True),
            nn.ReLU(),
            nn.Linear(ch // ratio, ch, bias=True),
        )
        # Sigmoid function
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_in):
        batch_size, num_channels, _, _ = f_in.shape
        # Formula from the paper equation (1)
        W = self.sigmoid(self.FC(self.gap(f_in).view(f_in.size(0), -1)))
        return torch.mul(W.view(batch_size, num_channels, 1, 1), f_in)

class AdaGN(nn.Module):
    ''' AdaGN recalibrates
        eye features with face appearance characteristics derived from face and Rects features.
        Recive input size and number of channels return shift and scale features
    '''
    def __init__(self, sz, ch):
        super(AdaGN, self).__init__()

        self.FC = nn.Linear(sz, 2 * ch)
        self.LeakyRelU = nn.LeakyReLU()

    def forward(self, f_in, block, ada_in):
        batch_size, num_channels, _, _ = f_in.shape

        # Scale and shift parameters for each channel as for equation (2) from the paper
        shape = self.LeakyRelU(self.FC(ada_in))
        shape = shape.view([batch_size, 2, num_channels, 1, 1])  
        scale = shape[:, 0, :, :, :]
        shift = shape[:, 1, :, :, :]

        f_gn = f_in.view(batch_size * block, -1)

        # Get mean and std of the batch
        std, mean = torch.std_mean(input=f_gn, dim=1, keepdim=True)
        # Normalize
        f_gn = (f_gn - mean) / (std + 1e-8)
        
        # Back to the original shape but with normal Group Normalization
        f_gn = f_gn.view(f_in.shape)

        f_out = (scale) * f_gn + shift
        return f_out


class Model(nn.Module):
    ''' Implement the main structure for the Adaptive Feature Fusion
    network.
    '''
    def __init__(self):
        super(Model, self).__init__()

        # Eyes and face model
        self.eyeModel = EyeModel()
        self.faceModel = FaceModel()

        self.conv = nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)
        self.se1 =SELayer(256, 16)   
        self.se2 = SELayer(64, 16)
        self.relu = nn.ReLU()
        self.agn = AdaGN(128, 64)

        # Fully conected layers for eyes and face and rects
        self.FCLayer = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )
        self.eyesFC = nn.Sequential(
            nn.Linear(3136, 128),
            nn.LeakyReLU(),
        )
        self.rectsFC = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 96),
            nn.LeakyReLU(),
            nn.Linear(96, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        self.loss_fn = nn.SmoothL1Loss().cuda()

    def training_step(self, data, device):
        output = self(data['leftEye'].to(device), 
                     data['rightEye'].to(device),
                     data['face'].to(device),
                     data['rects'].to(device))

        loss = self.loss_fn(output, data['label'].to(device)) * 4
        return loss

    def validation_step(self, data, device):
        output = self(data['leftEye'].to(device), 
                     data['rightEye'].to(device),
                     data['face'].to(device),
                     data['rects'].to(device)) 

        # Labels are the score which are the normalized position on screen
        labels = data['label'].to(device)
        # Resolution of the screen
        resolution = data['resolution'].to(device)
        return output, labels, resolution

    def forward(self, eyesLeft, eyesRight, faces, rects):

        # Calibration step
        outFace = self.faceModel(faces)
        outRect = self.rectsFC(rects)
        ada_in = torch.cat((outFace, outRect), 1)

        # Apply the eye model to the left eye
        outEyeL = self.eyeModel(eyesLeft, ada_in)
        # Apply the eye model to the left eye
        outEyeR = self.eyeModel(eyesRight, ada_in)
        # Concat both eyes
        outEyes = torch.cat((outEyeL, outEyeR), 1)

        # Apply SELayer
        outEyes = self.se1(outEyes)
        # Apply a convolutional layer
        outEyes = self.conv(outEyes)
        # Apply AdaGN
        outEyes = self.agn(outEyes, 8, ada_in)
        outEyes = self.relu(outEyes)
        outEyes = self.se2(outEyes)
        outEyes = outEyes.view(outEyes.size(0), -1)
        # Apply Fully Conected layer
        outEyes = self.eyesFC(outEyes)

        # Apply eyes, face and rects features
        f_out = torch.cat((outEyes, outFace, outRect), 1)
        return self.FCLayer(f_out)
