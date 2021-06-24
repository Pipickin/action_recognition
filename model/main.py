import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import torch
import preproc
from torch.utils.data import DataLoader
import model
import time
import os
import plot


def train(CNN_model, LSTM_model, train_loader, epoch, device, optimizer, criterion):
    CNN_model = CNN_model
    LSTM_model = LSTM_model
    CNN_model.train()
    LSTM_model.train()
    total_loss_sum = 0
    total_acc_sum = 0
    total = 0
    batches_loss = 0

    for batch, (X, label) in enumerate(train_loader):

        X, label = X.to(device), label.to(device).view(-1, )
        optimizer.zero_grad()
        output = LSTM_model(CNN_model(X))

        loss = criterion(output, label)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        # print(predicted.eq(label.view_as(predicted)).sum().item())
        total_acc_sum += predicted.eq(label.view_as(predicted)).sum().item()
        loss.backward()
        optimizer.step()
        total_loss_sum += loss.item()

        batches_loss += loss.item()
        if batch % 300 == 299:    # print every 300 mini-batches
            print('[%d, %d] Loss: %.5f' %
                  (epoch, batch + 1, batches_loss / 300))
            batches_loss = 0.0

    current_train_loss = total_loss_sum / len(train_loader)
    current_train_acc = total_acc_sum / total

    print('Train epoch: {}, Train size: {} \nLoss: {:.5f}, \tAccuracy: {:.2f}%'.format(
        epoch + 1, len(train_loader.dataset), current_train_loss, 100 * current_train_acc))

    return current_train_loss, current_train_acc


def test(CNN_model, LSTM_model, test_loader, device, criterion):
    CNN_model = CNN_model
    LSTM_model = LSTM_model
    CNN_model.eval()
    LSTM_model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for X, label in test_loader:
            X, label = X.to(device), label.to(device).view(-1, )
            output = LSTM_model(CNN_model(X))

            loss += criterion(output, label).item()
            _, predicted = output.max(1, keepdim=True)
            correct += predicted.eq(label.view_as(predicted)).sum().item()

    loss /= len(test_loader)
    acc = correct / len(test_loader)
    print('\nTest size = {:d}\nAverage loss: {:.5f}, \tAccuracy: {:.2f}%\n'.format(len(test_loader),
                                                                                   loss, 100 * acc))
    return loss, acc


if __name__ == '__main__':

    train_path = r'./data_ext/train'
    test_path = r'./data_ext/test'
    # train_path = r'./t/train'
    # test_path = r'./t/test'
    chpt_path = r'./chpt'

    PATH_CNN = r'weights/CNN_ext_4conv.pth'
    PATH_LSTM = r'weights/LSTM_ext_4conv.pth'

    final_plot_loss_path = 'Loss_ext_4conv.png'
    final_plot_acc_path = 'Accuracy_ext_4conv.png'

    train_folders, train_labels = preproc.get_folders_and_labels(train_path)
    test_folders, test_labels = preproc.get_folders_and_labels(test_path)

    batch_size = 1
    epochs = 20
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    transform = tf.Compose([tf.ToTensor(), tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_dataset = preproc.Dataset(train_path, train_folders, train_labels, transform)
    test_dataset = preproc.Dataset(test_path, test_folders, test_labels, transform)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=2)

    CNN_model = model.CNN(16).to(device)
    LSTM_model = model.LSTM(16, 256, 2, 2).to(device)

    total_params = list(CNN_model.parameters()) + list(LSTM_model.parameters())
    optimizer = torch.optim.Adam(total_params, weight_decay=0.0001, lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    # criterion = F.binary_cross_entropy()

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(1, epochs + 1):
        train_loss_epoch, train_acc_epoch = train(CNN_model, LSTM_model, train_dataloader, epoch, device, optimizer, criterion)
        test_loss_epoch, test_acc_epoch = test(CNN_model, LSTM_model, test_dataloader, device, criterion)

        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)
        train_acc.append(train_acc_epoch)
        test_acc.append(test_acc_epoch)

        if epoch % 5 == 0:
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            chpt_folder_path = os.path.join(chpt_path, dt)
            os.mkdir(chpt_folder_path)
            cnn_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_cnn.pth")
            lstm_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_lstm.pth")
            torch.save(CNN_model.state_dict(), cnn_path)
            torch.save(LSTM_model.state_dict(), lstm_path)

            loss_path = os.path.join(chpt_folder_path, 'Loss_%d.png' % epoch)
            acc_path = os.path.join(chpt_folder_path, 'Accuracy_%d.png' % epoch)
            plot.save_all_plots(train_loss, test_loss,
                                train_acc, test_acc,
                                loss_path, acc_path)

    torch.save(CNN_model.state_dict(), PATH_CNN)
    torch.save(LSTM_model.state_dict(), PATH_LSTM)

    plot.save_all_plots(train_loss, test_loss,
                        train_acc, test_acc,
                        final_plot_loss_path, final_plot_acc_path)

    print('*' * 100)
    print('Total train accuracy = ', str(sum(train_acc) / len(train_acc)))
    print('Total test accuracy = ', str(sum(test_acc) / len(test_acc)))
    print('Total train loss = ', str(sum(train_loss) / len(train_loss)))
    print('Total test loss = ', str(sum(test_loss) / len(test_loss)))
    print('*' * 100)
