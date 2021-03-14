# Dataset类的构建

Dataset类用于从指定文件夹中获取图片并对图片进行预处理然后返回处理后的图片以及标签信息。

Dataset类中都会有三个初始化方法：

1. `__init__`方法用于实现初始化参数，确定传入参数
2. `__getitem__`方法用于通过索引获取指定文件夹中的一张图片并进行预处理
3. `__len__`方法用于返回整个数据集的长度

```python
class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
```

下面为一个dataset示例：

```python
#读取文件夹中的图片进行处理
class ImageDataset(Dataset):
    def __init__(self,folder_path,img_size):
        self.files = sorted(glob.glob("%s/*.*"%folder_path)) #获取文件夹中所有文件组成一个list
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img,_ = image_to_square(img)
        img = resize(img,self.img_size)
        return img_path,img

    def __len__(self):
        return len(self.files)
```

