import torch
import torchvision.transforms as tf
import cv2
import os
import model


def predict(CNN_model, LSTM_model, folder_path, device):
    frames = get_frames_from_folder(folder_path)
    frames = frames.to(device)
    outputs = LSTM_model(CNN_model(frames))
    prediction = outputs.max(1, keepdim=True)[1]
    return prediction


def get_frames_from_folder(folder_path):
    transform = tf.Compose([tf.ToTensor(), tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    frames_names = os.listdir(folder_path)
    frames = []

    for frame_name in frames_names:
        frame_path = os.path.join(folder_path, frame_name)
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (256, 256))
        frame = transform(frame)
        frames.append(frame)

    frames = torch.stack(frames, dim=0).view(1, 60, 3, 256, 256)
    return frames


def get_answer(folder_path):
    return torch.tensor(int(folder_path[-1])).to('cuda')


def print_inference_acc(cnn_model, lstm_model, folder_path, device):
    cnn_model = cnn_model
    lstm_model = lstm_model
    prediction = []
    name_folder = []
    correct = 0
    num_folders = len(os.listdir(folder_path))
    for folder in os.listdir(folder_path):
        video_path = os.path.join(folder_path, folder)
        name_folder.append(folder)
        pred = predict(cnn_model, lstm_model, video_path, device).to('cuda')
        answer = get_answer(folder)
        correct += pred.eq(answer.view_as(pred)).sum().item()
        prediction.append(pred.item())
    accuracy = correct / num_folders
    predict_dict = dict(zip(name_folder, prediction))
    predict_dict = dict(sorted(predict_dict.items(), key=lambda k: int(k[0][:-2])))
    print('Accuracy = {}%'.format(round(accuracy * 100, 2)))
    print(predict_dict)


if __name__=="__main__":
    folder_path = r'./data_ext/inference/'
    cnn_path = r'./weights/CNN_ext.pth'
    lstm_path = r'./weights/LSTM_ext.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')

    CNN_model = model.CNN(5).to(device)
    LSTM_model = model.LSTM(5, 256, 2, 2).to(device)
    CNN_model.load_state_dict(torch.load(cnn_path), strict=False)
    LSTM_model.load_state_dict(torch.load(lstm_path), strict=False)
    CNN_model.eval()
    LSTM_model.eval()

    print_inference_acc(CNN_model, LSTM_model, folder_path, device)


    # prediction = []
    # name_folder = []
    # for folder in os.listdir(folder_path):
    #     video_path = os.path.join(folder_path, folder)
    #     name_folder.append(folder)
    #     local_predict = predict(CNN_model, LSTM_model, video_path, device)
    #     prediction.append(local_predict)
    # predict_dict = dict(zip(name_folder, prediction))
    # print(predict_dict)
