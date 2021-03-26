
import torch
import glob
import numpy as np

list_path = "/Users/lianxing/Desktop/server/PyTorch-YOLOv3.nosync/data/coco/trainvalno5k.txt"
with open(list_path, "r") as file:
    img_files = file.readlines()  # 读取图片位置
label_files = [
    path.replace("images", "labels").replace("png", "txt").replace("jpg", "txt")
    for path in img_files
]

label_path = label_files[0].rstrip()
boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
a=2

x = torch.randn(2,3)
y = torch.randn(2,3)
z = torch.stack((x,y),0)
print(z.size())