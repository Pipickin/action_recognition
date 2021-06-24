import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms as tf
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import dataset_ae_train as dat


def conv_relu(x, conv):
    return F.relu(conv(x))


class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, (3, 3), (2, 2), 1)
        self.conv2 = nn.Conv2d(256, 128, (3, 3), (2, 2), 1)
        self.conv3 = nn.Conv2d(128, 64, (3, 3), (2, 2), 1)
        self.conv4 = nn.Conv2d(64, 32, (3, 3), (2, 2), 1)
        self.fc = nn.Linear(8192, output_dim)
        # self.conv2 = nn.Conv2d

    def forward(self, image):
        x = conv_relu(image, self.conv1)
        x = conv_relu(x, self.conv2)
        x = conv_relu(x, self.conv3)
        x = conv_relu(x, self.conv4)
        x = self.fc(x.view(x.size(0), -1))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(input_dim, 8192)
        self.conv1 = nn.Conv2d(32, 64, (3, 3), (1, 1), 1)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), (1, 1), 1)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), (1, 1), 1)
        self.conv4 = nn.Conv2d(256, 3, (3, 3), (1, 1), 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d()

    def forward(self, input_vector):
        x = self.fc(input_vector)
        x = conv_relu(x.view(x.size(0), 32, 16, 16), self.conv1)
        x = self.upsample(x)
        x = conv_relu(x, self.conv2)
        x = self.upsample(x)
        x = conv_relu(x, self.conv3)
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
        return x


if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = r'../model/data_ext/train/1_1/'
    transform = tf.Compose([tf.ToTensor()])
    dataset = dat.Dataset(path, transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    encoder = Encoder(10).to(device)
    decoder = Decoder(10).to(device)
    encoder.eval()
    decoder.eval()

    im = next(iter(data_loader))
    print(im.shape)
    print(im)

    encoded = encoder(im.to(device))
    print(encoded.shape)
    print(encoded)

    decoded = decoder(encoded)
    print(decoded.shape)
    print(decoded)


