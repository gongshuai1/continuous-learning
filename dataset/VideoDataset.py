# !/usr/bin/env python
# =====================================================================================
#
# @Date: 2022-04-01 20:45
# @Author: gongshuai
# @File: VideoDataset.py
# @IDE: PyCharm
# @Func: Video Dataset
#
# =====================================================================================
import torch.utils.data
from cl.loader import extract_image_from_video, load_video_path


class VideoDataset(torch.utils.data.Dataset):
    """
    VideoDataset - specific defined for video dataset training
    """
    def __init__(self, data_dir, transforms, num_frames, loader=extract_image_from_video):
        """
        Initialize VideoDataset
        :param data_dir: the directory of train/validate/test videos
        :param transforms: augmentation
        :param num_frames: the number of frames will be extracted from each video
        :param loader: load function for each video
        """
        super(VideoDataset, self).__init__()
        self.data_paths = load_video_path(data_dir)
        self.transforms = transforms
        self.loader = loader
        self.num_frames = num_frames

    def __getitem__(self, index):
        """
        Get item from dataset
        :param index: index
        :return: a tensor of extracted frames from a video, dimension: [frames, channels, height, width]
        """
        data_path = self.data_paths[index]
        return self.loader(video_path=data_path, augmentation=self.transforms, num_frames=self.num_frames)

    def __len__(self):
        """c
        The size of video dataset
        :return: the size of video dataset
        """
        return len(self.data_paths)

    @staticmethod
    def collate_fn(batch):
        frames = torch.stack(batch, dim=0)
        return frames
