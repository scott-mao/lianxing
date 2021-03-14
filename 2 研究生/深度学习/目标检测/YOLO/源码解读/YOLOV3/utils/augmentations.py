import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1]) #将图片沿着w方向翻转
    targets[:, 2] = 1 - targets[:, 2]  #只有横坐标变了
    return images, targets

