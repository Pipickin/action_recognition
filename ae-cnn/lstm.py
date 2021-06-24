import torch.nn as nn
import dataset_ae_train as dat
import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import model


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x, None)
        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__=='__main__':
    train_path = r'/media/shchetkov/HDD/media/images/task3/train'
    test_path = r'/media/shchetkov/HDD/media/images/task3/test'
    # train_path = r'./t/train'
    # test_path = r'./t/test'

    PATH_ecnoder = r'chpt/2021_06_24-15_13_55_256/2021_06_24-15_13_55-10_encoder.pth'
    batch_size = 32
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    transform = tf.Compose([tf.ToTensor()])

    train_dataset = dat.Dataset(train_path, transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    latent_dim = 256
    encoder = model.Encoder(latent_dim).to(device)

    encoder.load_state_dict(torch.load(PATH_ecnoder))
    encoder.eval()

    image = next(iter(train_dataloader))
    output = encoder(image.to(device))
    print(output.shape)

    lstm = LSTM(256, 512, 2, 2).to(device)
    lstm.eval()

    lstm_output = lstm(output.view(1, batch_size, 256))
    print(lstm_output.shape)