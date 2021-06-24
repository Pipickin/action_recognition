from torch.utils import data
import cv2
import torch
import os
from model import Encoder
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import lstm


class Dataset(data.Dataset):
    def __init__(self, data_dir, folders, labels, encoder, transform):
        self.data_dir = data_dir
        self.folders = folders
        self.labels = labels
        self.transform = transform
        self.encoder = encoder

    def __len__(self):
        return len(self.folders)

    def get_images(self, path, selected_folder, transform):
        features = []

        for image_num in range(60):
            image = cv2.imread(os.path.join(path, selected_folder, '%d.jpg' % image_num), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            feature = self.encoder(image.view(1, 3, 256, 256).to('cuda'))
            features.append(feature)

        features = torch.stack(features, dim=0).view(60, 256)
        return features

    def __getitem__(self, idx):
        folder = self.folders[idx]
        images = self.get_images(self.data_dir, folder, self.transform)
        label = torch.LongTensor([self.labels[idx]])
        return images, label


def get_folders_and_labels(data_dir):
    folders = os.listdir(data_dir)
    labels = [int(folder_name[-1]) for folder_name in folders]
    return folders, labels


if __name__=='__main__':
    train_path = r'../model/data_ext/test'
    PATH_encoder = r'chpt/2021_06_24-15_13_55_256/2021_06_24-15_13_55-10_encoder.pth'

    transform = tf.Compose([tf.ToTensor()])
    encoder = Encoder(256).to('cuda')
    encoder.load_state_dict(torch.load(PATH_encoder))
    encoder.eval()

    train_folders, train_labels = get_folders_and_labels(train_path)
    train_dataset = Dataset(train_path, train_folders, train_labels, encoder, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    image, label = next(iter(train_dataloader))

    lstm = lstm.LSTM(256, 512, 2, 2).to('cuda')
    lstm.eval()

    output = lstm(image)
    print(output)