# !/usr/bin/env python
# =====================================================================================
#
# @Date: 2022-03-08 12:28
# @Author: gongshuai
# @File: loader.py
# @IDE: PyCharm
# @Func: load video data
#
# =====================================================================================
import os
import cv2
import time
import math
import torch
import numpy as np
import torchvision.transforms as transforms


def load_video_path(train_dir):
    """
    Load video path
    Args:
        train_dir: the path of training dataset
    Return:
        a list of all training videos' path
    """
    video_paths = []
    if os.path.exists(train_dir):
        for parent, dir_names, file_names in os.walk(train_dir):
            for file_name in file_names:
                video_paths.append(parent + "/" + file_name)
    return video_paths


def extract_image_from_video(video_path, augmentation, num_frames, interval=10):
    """
    Extract images from videos, we should
    Args:
        video_path: video path
        augmentation: augmentation
        num_frames: the total number of frames extracted from one video
        interval: interval
    Return:
        data_loader
    """
    frames = []

    # idx1 = video_path.rfind('/')
    # idx2 = video_path.rfind('.')
    # save_path = './dataset/' + video_path[idx1+1: idx2]
    #
    # if os.path.exists(save_path):
    #     pass
    # else:
    #     os.mkdir(save_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('FPS:{:.2f}'.format(fps))
    rate = cap.get(5)  # FPS
    frame_num = cap.get(7)  # total frame num
    duration = frame_num / rate
    print('video total time:{:.2f}s'.format(duration))


    # height, width = 1080, 1920
    cnt = 0
    num = 0
    total_images = frame_num // interval
    print('Total images:{:.0f}'.format(total_images))

    ts = time.time()
    while cap.isOpened() and num < num_frames:
        ret, image = cap.read()
        if ret:
            cnt += 1
            if cnt % interval == 0:
                num += 1
                image = torch.from_numpy(image)
                image = image.float()
                image = image.permute(2, 0, 1)  # (channels, height, width)
                image = augmentation(image)
                frames.append(image)
                # cv2.imwrite(save_path + '/%07d.jpg' % num, image)
                # remain_image = total_images - num
                # print('Processing %07d.jpg, remain images: %d' % (num, remain_image))
        else:
            break
        if cv2.waitKey(1) & 0xff == 27:
            break
    te = time.time()
    cap.release()
    cv2.destroyAllWindows()

    # The video is too short to get enough frames,
    # we reversal the extracted frames and append them to the end of the extracted frames
    # until the total number of frames is enough
    if num < num_frames:
        image_group = []
        num = 0
        length = len(frames)
        while num < num_frames:
            rounds = num // length
            idx = num % length
            if rounds % 2 == 0:
                image_group.append(frames[idx])
            else:
                image_group.append(frames[length - idx - 1])
            num += 1
    else:
        image_group = frames
    print('Process total time:{:.2f}'.format(te - ts))
    return torch.stack(image_group)  # (frames, channels, height, width)


def load_data(train_dir, batch_size, num_frames, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.CenterCrop(2048),
        transforms.Resize(512),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.ToTensor(),
        normalize
    ])


    # Load video paths
    video_paths = np.array(load_video_path(train_dir))
    idx = np.arange(len(video_paths))
    if shuffle:
        np.random.shuffle(idx)
    video_paths = video_paths[idx]
    batchs = math.ceil(len(video_paths) / batch_size)
    for i in range(batchs):
        batch_video_paths = video_paths[(i * batch_size): min((i + 1) * batch_size, len(video_paths))]
        mini_batch = []
        for video_path in batch_video_paths:
            mini_batch.append(extract_image_from_video(video_path, augmentation, num_frames, interval=10))
        yield torch.stack(mini_batch)  # (N, frames, channels, height, width)


if __name__ == '__main__':
    # training_data_dir = 'F:\papers\\video\dataset'
    # load_data(training_data_dir, batch_size=2, shuffle=True)
    video_path = 'F:\papers\\video\dataset\\animals\\n010003.mp4'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.CenterCrop(2048),
        transforms.Resize((512, 512)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.ToTensor(),
        normalize
    ])
    frames = extract_image_from_video(video_path, augmentation, 50)
    print(f'frames.shape = {frames.shape}')
