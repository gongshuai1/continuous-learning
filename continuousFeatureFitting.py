# !/usr/bin/env python
# =====================================================================================
#
# @Date: 2022-02-24 21:00
# @Author: gongshuai
# @File: continuousFeatureFitting.py
# @IDE: PyCharm
# @Func: fitting continuous features
#   1.extract feature from uniform continuous frames - to determine the gap of two consecutive frames
#   2.fit continuous features - to determine fitting method
#
# =====================================================================================
import os.path
import time
import cv2
import torch
import torchvision.transforms as transforms
import torchvision
from vit_pytorch import ViT
from PIL import Image
from torchvision import models


def extract_image_from_video(video_path, augmentation, interval=10):
    """
    Extract images from video
    :param augmentation: augmentation
    :param video_path: video path
    :param interval: interval
    :return:
    """
    image_group = []

    idx1 = video_path.rfind('/')
    idx2 = video_path.rfind('.')
    save_path = './dataset/' + video_path[idx1+1: idx2]

    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

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
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            cnt += 1
            if cnt % interval == 0:
                num += 1
                image = torch.from_numpy(image)
                image = image.float()
                image = image.permute(2, 0, 1)  # (channels, height, wight)
                image = augmentation(image)
                image_group.append(image)
                # cv2.imwrite(save_path + '/%07d.jpg' % num, image)
                remain_image = total_images - num
                print('Processing %07d.jpg, remain images: %d' % (num, remain_image))
        else:
            break
        if cv2.waitKey(1) & 0xff == 27:
            break
    te = time.time()
    cap.release()
    cv2.destroyAllWindows()
    print('Process total time:{:.2f}'.format(te - ts))
    return torch.stack(image_group)  # (frames, channels, height, weight)


def fitting():
    # 1.Load dataset
    # example video about Shiba Inu(柴犬) from:https://www.pexels.com/zh-cn/video/4503918/
    video_path = 'dataset/shiba_Inu.mp4'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.CenterCrop(2048),
        transforms.Resize(512),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.ToTensor(),
        normalize
    ])
    image_group = extract_image_from_video(video_path, augmentation)

    # 2.Load pretrained model - resNet or ViT
    # vit_model = ViT(image_size=512, patch_size=32, dim=1024, depth=6, heads=16, mlp_dim=2048)
    res50 = models.resnet50(pretrained=True)  # Input: (N, C, H, W)

    # 3.Extract feature
    feature_maps = res50(image_group)
    print('feature_maps = ' + str(feature_maps))

    # 4.Fitting continuous features
    compute_loss(feature_maps)
    pass


def compute_loss(feature_map):
    """
    Compute loss: cluster loss + continuous loss
    :param feature_map: feature map
    :return: loss
    """
    # cluster loss
    # define mean feature as cluster center
    mean_feature = torch.sum(feature_map, dim=0)/len(feature_map)
    cluster_loss = torch.sum(torch.norm(feature_map - mean_feature, p=2, dim=1))
    print('cluster_loss = ' + str(cluster_loss))

    # continuous loss
    idx = torch.arange(0, len(feature_map)-2)
    mid_features = (feature_map[idx] + feature_map[idx+2])/2
    continuous_loss = torch.sum(torch.norm(mid_features - feature_map[idx+1], p=2, dim=1))
    print('continuous_loss = ' + str(continuous_loss))


if __name__ == '__main__':
    fitting()
    # arr = []
    # arr.append(torch.ones((5, 5, 3)))
    # arr.append(torch.zeros((5, 5, 3)))
    # arr = torch.Tensor([item.numpy() for item in arr])
    # print('arr = ' + str(arr))

    # a = torch.rand((5, 10, 128))
    # idx = torch.arange(0, len(a[1]) - 2)
    # mid_features = (a[:, idx, :] + a[:, idx+2, :])/2
    # continuous_loss = torch.sum(torch.norm(mid_features - a[:, idx + 1, :], p=2, dim=1), dim=1)
    # print('continuous_loss = ' + str(continuous_loss))

    # mean_feature = torch.sum(a, dim=1)/len(a[1])
    # mean_feature = torch.unsqueeze(mean_feature, dim=1)
    # cluster_loss = torch.sum(torch.norm(a - mean_feature, p=2, dim=2), dim=1)
    # print('cluster_loss = ' + str(cluster_loss))
