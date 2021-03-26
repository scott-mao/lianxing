from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import os
import torch
from augmentations import horisontal_flip
import random
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from utils import *
#目的：建立dataset类，从文件夹中读取图片
#图片尺寸不一，需要将图片缩放到416*416


#首先将图片填充变成正方形
def pad_to_square(img,pad_value=0):
    '''
    使得图片变成正方形
    :param img: 输入为图片张量 c*h*w
    :param pad_value: 需要填充的值
    :return: 返回正方形图片
    '''
    c,h,w = img.shape
    dim_diff = np.abs(h-w)
    pad_value1,pad_value2 = dim_diff//2,dim_diff-dim_diff//2
    pad = (pad_value1,pad_value2,0,0) if h>w else (0,0,pad_value1,pad_value2)
    img = F.pad(img,pad,"constant",pad_value)
    # img = F.pad(image,(pad_value1,pad_value2),"reflect") 利用镜面映射填充
    return img,pad

def resize(image,size):
    '''
    将图片缩放到一定尺寸
    :param image: 输入为张量 c*h*w
    :return: 缩放后的图片张量
    '''
    #由于默认输入为 batch*c*h*w形式，所以先进行升维
    image = F.interpolate(image.unsqueeze(0),size=size,mode='nearest').squeeze(0)
    return image

#读取文件夹中的图片进行处理
class ImageDataset(Dataset):
    def __init__(self,folder_path,img_size):
        self.files = sorted(glob.glob("%s/*.*"%folder_path)) #获取文件夹中所有文件组成一个list
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img,_ = pad_to_square(img)
        img = resize(img,self.img_size)
        return img_path,img

    def __len__(self):
        return len(self.files)

folder_path = "/Users/lianxing/Desktop/server/PyTorch-YOLOv3/data/samples"
image_test = ImageDataset(folder_path,412)
a=2


class ListDataset(Dataset):
    def __init__(self,list_path,img_size=416,augment=True,mutiscale=True,normalized_labels=True):
        with open(list_path,"r") as file:
            self.img_files = file.readlines() #读取图片位置
        self.label_files = [
            path.replace("images","labels").replace("png","txt").replace("jpg","txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.augment = augment
        self.mutiscale = mutiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, item):
        img_path = self.img_files[item % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # if len(img.shape != 3 ):
        #     img = img.unsqueeze(0)
        #     img = img.expand((3, img.shape[1:]))

        _,h,w = img.shape

        img,pad = pad_to_square(img,0)
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        _,padded_h,padded_w= img.shape

        #Label
        label_path = self.label_files[item % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)

            return img_path, img, targets

    def collact_fn(self,batch):
        paths,imgs,targets = list(zip(*batch))
        #去除掉空标记
        targets = [boxes for boxes in targets if boxes is not None]
        #为每一个boxes建立索引
        for i,boxes in enumerate(targets):
            boxes[:,0] = i
        targets = torch.cat(targets, 0) #不知道具体作用

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)







def image_show(image):
    '''
    将图片显示出来
    :param image:传入为image张量，3*h*w
    '''
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    plt.imshow(img)
    plt.show()# 显示图片





import matplotlib.pyplot as plt

def image_show(image):
    '''
    将图片显示出来
    :param image:传入为image张量，3*h*w
    '''
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    plt.imshow(img)
    plt.show()# 显示图片






#
# img_path = "/Users/lianxing/Desktop/server/PyTorch-YOLOv3/data/custom/images/train.jpg"
# image=Image.open(img_path)
# image = transforms.ToTensor()(image)
# image,_ = pic_to_square(image,0)
# image_show(image)
# a=2

show_ListDataset = ListDataset('/Users/lianxing/Desktop/server/PyTorch-YOLOv3.nosync/data/coco/trainvalno5k.txt')
# 显示数据组织
print('img_path:')  # 路径
print(show_ListDataset[111][0])
print('img     :')  # 图片
print(show_ListDataset[111][1])
print('label   :')  # 标签
print(show_ListDataset[111][2])

getImg2 = np.transpose(show_ListDataset[111][1], (1, 2, 0))
imshow(getImg2)

fig,ax = plt.subplots(1)
ax.imshow(getImg2)

cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in range(20)]
unique_labels = show_ListDataset[111][2]
n_cls_preds = len(unique_labels)
bbox_colors = random.sample(colors,n_cls_preds)
classes = load_classes("/Users/lianxing/Desktop/server/PyTorch-YOLOv3.nosync/data/coco.names")




a=2