import torch
import torchvision.transforms as tf
import cv2
import os
import model
from lstm import LSTM


def predict(encoder, LSTM_model, folder_path, device):
    frames = get_frames_from_folder(folder_path, encoder)
    frames = frames.to(device)
    outputs = LSTM_model(frames)
    prediction = outputs.max(1, keepdim=True)[1]
    return prediction


def get_frames_from_folder(folder_path, encoder):
    transform = tf.Compose([tf.ToTensor()])
    frames_names = os.listdir(folder_path)
    frames = []

    for frame_name in frames_names:
        frame_path = os.path.join(folder_path, frame_name)
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        frame = transform(frame)
        feature = encoder(frame.view(1, 3, 256, 256).to('cuda'))
        frames.append(feature)

    frames = torch.stack(frames, dim=0).view(1, 60, 256)
    return frames


def get_answer(folder_path):
    return torch.tensor(int(folder_path[-1])).to('cuda')


def print_inference_acc(encoder, lstm_model, folder_path, device):
    encoder = encoder
    lstm_model = lstm_model
    prediction = []
    name_folder = []
    correct = 0
    num_folders = len(os.listdir(folder_path))
    for folder in os.listdir(folder_path):
        video_path = os.path.join(folder_path, folder)
        name_folder.append(folder)
        pred = predict(encoder, lstm_model, video_path, device).to('cuda')
        answer = get_answer(folder)
        correct += pred.eq(answer.view_as(pred)).sum().item()
        prediction.append(pred.item())
    accuracy = correct / num_folders
    predict_dict = dict(zip(name_folder, prediction))
    predict_dict = dict(sorted(predict_dict.items(), key=lambda k: int(k[0][:-2])))
    print('Accuracy = {}%'.format(round(accuracy * 100, 2)))
    print(predict_dict)


if __name__=="__main__":
    folder_path = r'../model/data_ext/inference/'
    encoder_path = r'./chpt/2021_06_24-15_13_55_256/2021_06_24-15_13_55-10_encoder.pth'
    lstm_path = r'./chpt/2021_06_24-19_15_32_lstm_256/2021_06_24-19_15_32-10_lstm.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    encoder = model.Encoder(256).to(device)
    LSTM_model = LSTM(256, 512, 2, 2).to(device)
    encoder.load_state_dict(torch.load(encoder_path), strict=False)
    LSTM_model.load_state_dict(torch.load(lstm_path), strict=False)
    encoder.eval()
    LSTM_model.eval()

    print_inference_acc(encoder, LSTM_model, folder_path, device)
