# !/usr/bin/env python
# =====================================================================================
#
# @Date: 2022-03-08 10:11
# @Author: gongshuai
# @File: loss.py
# @IDE: PyCharm
# @Func: loss function consists of cluster loss and continuous loss
#
# =====================================================================================
import torch
from torch import Tensor
from torch.nn.modules import Module


class ClusterLoss(Module):
    """
    Based on the assumption that good feature representations of continuous frames from a video
    using same encoder should contains object's appearance feature in the video,
    we propose cluster loss to denote object appearance loss.
    In other word, features from continuous frames should as close as possible in latent space.
    """
    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, input_group: Tensor) -> Tensor:
        """
        As a simple implementation, we define mean feature as cluster center of each video
        Args：
            input_group: a tensor of continuous features for some video in a mini-batch
        Shape:
            input_group: `(N, f, d)` where `N` is the number of videos in a mini batch and
                `f` is the number of frames in per video and `d` is dimension of feature extracted from each frame
        Return:
            a tensor of cluster loss for each video
        """
        # Define mean feature as cluster center
        mean_feature = torch.sum(input_group, dim=1)/len(input_group[1])
        mean_feature = torch.unsqueeze(mean_feature, dim=1)
        cluster_loss = torch.sum(torch.norm(input_group - mean_feature, p=2, dim=2), dim=1)
        return cluster_loss


class ContinuousLoss(Module):
    """
    Based on the assumption that features extracted from continuous frames should sequential in latent space,
    we propose continuous loss to denote the smooth degree of features extracted from continuous frames in latent space
    """
    def __init__(self):
        super(ContinuousLoss, self).__init__()

    def forward(self, input_group: Tensor) -> Tensor:
        """
        As a simple implement, we use three consecutive features as a group,
        and compute the distance from the second feature to the straight line
        defined by the first feature and the third feature in latent space
        Args:
            input_group: a tensor of continuous features for some video in a mini-batch
        Shape:
            input_group: `(N, f, d)` where `N` is the number of videos in a mini batch and
                `f` is the number of frames in per video and `d` is dimension of feature extracted from each frame
        Return:
            a tensor of continuous loss for each video
        """
        idx = torch.arange(0, len(input_group[1]) - 2)
        mid_features = (input_group[:, idx, :] + input_group[:, idx + 2, :]) / 2
        continuous_loss = torch.sum(torch.norm(mid_features - input_group[:, idx + 1, :], p=2, dim=1), dim=1)
        return continuous_loss


class ConsistentContinuousLoss(Module):
    def __init__(self):
        super(ConsistentContinuousLoss, self).__init__()
        self.cluster_loss = ClusterLoss()
        self.continuous_loss = ContinuousLoss()

    def forward(self, input_group: Tensor):
        clu_loss = self.cluster_loss(input_group)
        con_loss = self.continuous_loss(input_group)
        return clu_loss, con_loss
