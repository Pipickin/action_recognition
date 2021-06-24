import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    """" Class for convnet part to convert the image to the features. """
    def __init__(self, output_num=2):
        """"Initialize convnet. There are 3 cnnBlock, 2 fully connected layers."""
        super(CNN, self).__init__()

        self.conv1 = self.cnnBlock(3, 32, 3, 1, 1, 2)
        self.conv2 = self.cnnBlock(32, 64, 3, 1, 1, 2)
        self.conv3 = self.cnnBlock(64, 128, 3, 1, 1, 2)
        self.conv4 = self.cnnBlock(128, 256, 3, 1, 1, 2)
        self.fc1 = nn.Linear(65536, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, output_num)

    def cnnBlock(self, input_size, output_size, kernel_size=3, padding=1, stride=1, pooling=2):
        """"Create cnnBlock which consist of Conv2d, BatchNorm2d, activation function and MaxPool2d"""
        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, (kernel_size, kernel_size), padding=padding, stride=stride),
            nn.BatchNorm2d(output_size, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d((pooling, pooling))
        )
        return block

    def forward(self, images):
        outputs_images = []
        for batch in range(images.shape[0]):
            image = images[batch, :, :, :, :]
            x = self.conv1(image)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.drop2(x)
            output_image = self.fc3(x)
            outputs_images.append(output_image)
        outputs = torch.stack(outputs_images)
        return outputs


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x, None)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model_CNN = CNN(5).to(device)
    model_LSTM = LSTM(5, 256, 2, 2).to(device)
    path = './data/test/1_1/0.jpg'
    path2 = './data/test/1_1/1.jpg'
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    img3 = image2.copy()

    tensors = []
    transform = tf.Compose([tf.ToTensor(), tf.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
    tensors.append(transform(image))
    tensors.append(transform(image2))
    tensors.append(transform(img3))

    tensors2 = []
    tensors2.append(transform(image))
    tensors2.append(transform(image2))
    tensors2.append(transform(img3))

    tensors = torch.stack(tensors, dim=0)
    # tensors2 = torch.stack(tensors2, dim=0)
    # tensors3 = torch.stack([tensors, tensors2], dim=0)
    print(tensors.shape)

    pred = model_CNN(tensors.float().to(device))
    print(pred.shape)
    predict_lstm = model_LSTM(pred.view(1, 3, 5))
    print(predict_lstm.shape)


