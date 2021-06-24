import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import model
import time
import os
import dataset_ae_train
import matplotlib.pyplot as plt


def save_loss(train_loss, test_loss, save_path):
    fig, ax = plt.subplots()
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.savefig(save_path)


def train(encoder, decoder, train_loader, epoch, device, optimizer, criterion):
    encoder = encoder
    decoder = decoder
    encoder.train()
    decoder.train()
    total_loss_sum = 0
    batches_loss = 0

    for batch, X in enumerate(train_loader):

        X = X.to(device)
        optimizer.zero_grad()
        output = decoder(encoder(X))

        loss = criterion(output, X)

        loss.backward()
        optimizer.step()
        total_loss_sum += loss.item()

        batches_loss += loss.item()
        if batch % 500 == 499:
        # if True:
            print('[%d, %d] Loss: %.5f' % (epoch, batch + 1, batches_loss / 500))
            batches_loss = 0.0

    current_train_loss = total_loss_sum / len(train_loader)

    print('Train epoch: {}, Train size: {} \nLoss: {:.5f}'.format(
        epoch, len(train_loader.dataset), current_train_loss))

    return current_train_loss


def test(encoder, decoder, test_loader, device, criterion):
    encoder = encoder
    decoder = decoder
    encoder.eval()
    decoder.eval()
    loss = 0
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)

            output = decoder(encoder(X))

            loss += criterion(output, X).item()

    loss /= len(test_loader)
    print('\nTest size = {:d}\nAverage loss: {:.5f}\n'.format(len(test_loader.dataset), loss))

    return loss


if __name__ == '__main__':

    train_path = r'/media/shchetkov/HDD/media/images/task3/train'
    test_path = r'/media/shchetkov/HDD/media/images/task3/test'
    # train_path = r'./t/train'
    # test_path = r'./t/test'
    chpt_path = r'./chpt'

    PATH_ecnoder = r'weights/encoder.pth'
    PATH_decoder = r'weights/decoder.pth'
    batch_size = 32
    epochs = 20
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    transform = tf.Compose([tf.ToTensor()])

    train_dataset = dataset_ae_train.Dataset(train_path, transform)
    test_dataset = dataset_ae_train.Dataset(test_path, transform)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    latent_dim = 256
    encoder = model.Encoder(latent_dim).to(device)
    decoder = model.Decoder(latent_dim).to(device)

    total_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(total_params, weight_decay=0.0001, lr=0.0001)
    criterion = nn.MSELoss()
    # criterion = F.binary_cross_entropy()

    train_loss = []
    test_loss = []

    for epoch in range(1, epochs + 1):
        train_loss_epoch = train(encoder, decoder, train_dataloader, epoch, device, optimizer, criterion)
        test_loss_epoch = test(encoder, decoder, test_dataloader, device, criterion)

        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)

        if epoch % 5 == 0:
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            chpt_folder_path = os.path.join(chpt_path, dt)
            os.mkdir(chpt_folder_path)
            cnn_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_encoder.pth")
            lstm_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_decoder.pth")
            torch.save(encoder.state_dict(), cnn_path)
            torch.save(decoder.state_dict(), lstm_path)

            loss_path = os.path.join(chpt_folder_path, 'Loss_%d.png' % epoch)
            save_loss(train_loss, test_loss, loss_path)

    torch.save(encoder.state_dict(), PATH_ecnoder)
    torch.save(decoder.state_dict(), PATH_decoder)

    final_loss_path = 'Loss.png'
    save_loss(train_loss, test_loss, final_loss_path)

    print('*' * 100)
    print('Total train loss = ', str(sum(train_loss) / len(train_loss)))
    print('Total test loss = ', str(sum(test_loss) / len(test_loss)))
    print('*' * 100)
