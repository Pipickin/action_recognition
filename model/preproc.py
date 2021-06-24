from torch.utils import data
import cv2
import torch
import os


class Dataset(data.Dataset):
    def __init__(self, data_dir, folders, labels, transform):
        self.data_dir = data_dir
        self.folders = folders
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def get_images(self, path, selected_folder, transform):
        images = []

        for image_num in range(60):
            image = cv2.imread(os.path.join(path, selected_folder, '%d.jpg' % image_num), cv2.IMREAD_COLOR)
            image = transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        return images

    def __getitem__(self, idx):
        folder = self.folders[idx]
        images = self.get_images(self.data_dir, folder, self.transform)
        label = torch.LongTensor([self.labels[idx]])
        return images, label


def get_folders_and_labels(data_dir):
    folders = os.listdir(data_dir)
    labels = [int(folder_name[-1]) for folder_name in folders]
    return folders, labels

