import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')
parser.add_argument('--video_path')
parser.add_argument('--shape_resize', default=(256, 256))
parser.add_argument('--label')
parser.add_argument('--frames_per_folder', default=60)
parser.add_argument('--save_folder')
parser.add_argument('--all_folder', default=False)

opt = parser.parse_args()


def video2frames(cap, last_free_fold):

    max_folders = get_max_folders(cap)
    for folder_num in range(max_folders):
        folder_path = os.path.join(save_folder, str(last_free_fold + folder_num) + ending)
        os.mkdir(folder_path)
        current_frame = 0
        for frame_index in range(folder_num * frames_per_folder, (folder_num + 1) * frames_per_folder):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            _, frame = cap.read()
            frame = cv2.resize(frame, shape_resize)
            cv2.imwrite(os.path.join(folder_path, str(current_frame) + '.jpg'), frame)
            current_frame += 1


def get_max_folders(capture):
    max_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return max_frames // frames_per_folder


def get_label(name):
    return name.split('_')[-1]


if __name__ == '__main__':
    video_path = opt.video_path
    save_folder = opt.save_folder
    shape_resize = opt.shape_resize
    ending = '_' + opt.label
    frames_per_folder = int(opt.frames_per_folder)

    folders = os.listdir(save_folder)
    if len(folders) == 0:
        last_free_folder = 0
    else:
        last_folder = max([int(folder.split('_')[0]) for folder in folders])
        last_free_folder = last_folder + 1

    if opt.all_folder:
        videos = os.listdir(video_path)
        for video_name in videos:
            print(video_name)
            single_video_path = os.path.join(video_path, video_name)
            video = cv2.VideoCapture(single_video_path)
            video2frames(video, last_free_folder)
            last_free_folder += get_max_folders(video)

    else:
        video = cv2.VideoCapture(video_path)
        video2frames(video, last_free_folder)


