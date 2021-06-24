from torch.utils import data
import cv2
import torch
import os


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.images_names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def get_image(self, img_path):
        image = cv2.imread(os.path.join(self.data_dir, img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image_name = self.images_names[idx]
        image = self.get_image(image_name)
        return image


