import torch
import torch.nn as nn
from utils.utils import *
from utils.parse_config import *
import torch.nn.functional as F

# https://blog.csdn.net/weixin_36714575/article/details/113870584
def creat_module(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = hyperparams['channels']
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == 'convolutional':
            bn = int(module_defs["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad = (kernel_size-1)//2 #same-padding
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels = output_filters[-1],
                    out_channels = filters,
                    kernel_size = kernel_size ,
                    stride = stride,
                    padding = pad,
                    bias = not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}",nn.BatchNorm2d(filters,momentum=0.9,eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}",nn.LeakyReLU(0.1))

        elif module_def["type"] == 'maxpool':
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            pad = (kernel_size-1)//2
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}",nn.ZeroPad2d((0,1,0,1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=pad)
            modules.add_module(maxpool)

        elif module_def["type"] == 'upsample':  #上采样
            upsample = Upsample(scale_factor=module_def["stride"],mode="nearest")
            modules.add_module(f"upsample_{module_i}",upsample)

        elif module_def["type"] == 'route':
            # layers = [int(x) for x in module_def["layers"].split(",")]
            # filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}",Emptylayer)

        elif module_def["type"] == 'shortcut':
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}",)

        elif module_def["type"] == 'yolo':
            anchor_idx = [int(x) for x in module_def['mask'].split(",")]
            anchors = [int(x) for x in module_def['anchors'].split(",")]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in anchor_idx]
            num_classes = module_def["classes"]











            modules.add_module(f"maxpool_{module_def}",nn.Maxpool2d())



#上采样
class Upsample(nn.Module):
    def __init__(self,scale_factor,mode = "nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self,x):
        x = F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)


class Emptylayer(nn.Module):
    def __init__(self):
        super(Emptylayer,self).__init__()






class YOLOLayer(nn.Module):
    """
    检测层，输入为之前模型输出 B*255*N*N
    """

    def __init__(self,anchors,num_classes,img_dim=416):
        super(YOLOLayer,self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.mse_loss = nn.MSELoss



        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self,grid_size,cuda=True):
        '''
        之前计算的xyhw是相对于grid而言的，现在要相对于图片
        :param grid_size: 网格数
        :param cuda: 是否GPU
        :return: 需要进行偏移的值和缩放后的xyhw
        '''
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim/g

        self.grid_x = torch.arange(g).repeat(g,1).view([1,1,g,g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g,1).view([1,1,g,g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride,a_h / self.stride) for a_w,a_h in self.anchors ])
        self.anchors_w = self.scaled_anchors[:,0:1].view([1,self.num_anchors,1,1])
        self.anchors_h = self.scaled_anchors[:,1:2].view([1,self.num_anchors,1,1])







    def forward(self,x,targets = None,img_dim=None):
        """
        前向传播输出检测框的值以及损失
        :param x: 前面卷积神经网络的输出 B*255*N*N 其中255为（80类+4个坐标预测+1置信度）*3个检测anchor
        :param targets: 检测框的GT
        :param img_dim: 图片的尺寸
        :return: 检测框的坐标以及损失
        """
        #使用cude GPU
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor


        self.img_dim = img_dim
        num_samples = x[0]  #获取batch数量
        grid_size = x[2]  #featuremap的尺寸

        #将输入x B*255*N*N --> B*num_anchors(3)*N*N*detection(5+80)
        prediction = (
            x.view(num_samples,self.num_anchors,(5+self.num_classes),grid_size,grid_size)
            .permute(0,1,3,4,2)
            .contiguous()
        )

        #得到检测框的值 (x,y,w,h)+预测置信度+预测类，此时预测的xyhw为相对量，相对于一个grid而言的
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        pred_conf = torch.sigmoid(prediction[...,4])
        pred_cls = torch.sigmoid(prediction[...,5:])

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size,cuda=x.is_cuda)

        #添加偏移量以及缩放参数后得到的BBox,此时xyhw仍为单位相对量，只不过为相对于原图尺寸而言
        pred_boxes = FloatTensor(prediction[...,:4].shape)
        pred_boxes[...,0] = x.data + self.grid_x #添加坐标偏移量
        pred_boxes[...,1] = y.data + self.grid_y
        prediction[...,2] = torch.exp(w.data) * self.anchors_w
        prediction[...,3] = torch.exp(h.data) * self.anchors_h

        #计算对应于原图的xyhw坐标真实值  B*(num_anchores*N*N)*(4+1+80)
        output = torch.cat(
            (
                pred_boxes.view(num_samples,-1,4) * self.stride,
                pred_conf.view(num_samples,-1,1),
                pred_cls.view(num_samples,-1,self.num_classes),
            ),
            -1
        )

        if targets is None:
            return output,0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask,tx,ty,tw,th,tcls,tconf = bulid_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
        loss_x = self.mse_loss(x,tx)
        loss_y = self.mse_loss(y,ty)
        loss_h = self.mse_loss(x,tx)
        loss_w = self.mse_loss(x,tx)
        loss_cls = self.mse_loss(tcls,tx)






















class Darknet(nn.module)

    def __init__(self, config_path, img_size = 416):
        super(Darknet,self).__init__()
        self.module_defs = parse_model_config(config_path)



    def forward(self,x,targets=None):
