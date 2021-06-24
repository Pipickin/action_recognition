import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
import dataset_ae_train as dat
import model

test_path = r'/media/shchetkov/HDD/media/images/task3/train'
PATH_ecnoder = r'chpt/2021_06_24-15_13_55_256/2021_06_24-15_13_55-10_encoder.pth'
PATH_decoder = r'chpt/2021_06_24-15_13_55_256/2021_06_24-15_13_55-10_decoder.pth'
transform = tf.Compose([tf.ToTensor()])

batch_size = 1
device = 'cuda'

test_dataset = dat.Dataset(test_path, transform)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

latent_dim = 256
encoder = model.Encoder(latent_dim).to(device)
decoder = model.Decoder(latent_dim).to(device)

encoder.load_state_dict(torch.load(PATH_ecnoder), strict=False)
decoder.load_state_dict(torch.load(PATH_decoder), strict=False)

encoder.eval()
decoder.eval()

batch = next(iter(test_dataloader))
source = batch.numpy().copy()
r2 = source[0][0, :, :]
g2 = source[0][1, :, :]
b2 = source[0][2, :, :]

z2 = cv2.merge([b2, g2, r2])
cv2.imshow('source_bgr', z2)

output = decoder(encoder(batch.to(device))).to('cpu').detach().numpy()
image_to_show = output[0]
print(image_to_show.shape)

r = image_to_show[0, :, :]
g = image_to_show[1, :, :]
b = image_to_show[2, :, :]

z = cv2.merge([b, g, r])
cv2.imshow('z', z)
cv2.waitKey(0)




