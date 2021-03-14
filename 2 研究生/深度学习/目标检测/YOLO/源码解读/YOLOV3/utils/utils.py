import torch


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bulid_targets(pred_boxes,pred_cls,target,anchors,ignore_thres):
    '''
    1.找到最好的检测框anchor，并计算其对应pred_box与target的IOU
    2.获取target在grid中的位置
    3.获得正例，负例用于loss的计算
    :param pred_boxes:
    :param pred_cls:
    :param target:
    :param anchors:
    :param ignore_thres:
    :return:
    '''

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  #batch_num
    nA = pred_boxes.size(1) #achors_num
    nC = pred_cls.size(-1) #classes_num
    nG = pred_boxes.size(2) #grid_num = nG*nG

    obj_mask = ByteTensor(nB,nA,nG,nG).fill_(0)
    noobj_mask = ByteTensor(nB,nA,nG,nG).fill_(1)
    class_mask = ByteTensor(nB,nA,nG,nG).fill_(0)
    iou_scores = ByteTensor(nB,nA,nG,nG).fill_(0)
    tx = FloatTensor(nB,nA,nG,nG).fill_(0)
    ty = FloatTensor(nB,nA,nG,nG).fill_(0)
    tw = FloatTensor(nB,nA,nG,nG).fill_(0)
    th = FloatTensor(nB,nA,nG,nG).fill_(0)
    tcls = FloatTensor(nB,nA,nG,nG,nC).fill_(0)


    target_boxes = target[:,2:6] * nG
    gxy = target_boxes[:,:2]
    gwh = target_boxes[:,2:]
    ious = torch.stack([bbox_wh_iou(anchor,gwh) for anchor in anchors]) #A*T
    best_ious,best_n = ious.max(0)

    #获取target中的值
    b,target_labels = target[:,:2].long().t()
    gx,gy = gxy.t()
    gw,gh = gwh.t()
    gi,gj = gxy.long().t() #标记哪一个grid含有目标

    obj_mask[b,best_n,gj,gi] = 1
    noobj_mask[b,best_n,gj,gi] = 0

    for i,anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i],anchor_ious>ignore_thres,gj[i],gi[i]] =0 #第i张图片找出不需要计算loss的anchor

    tx[b,best_n,gj,gi] = gx - gx.floor()  #相对于grid左上角的偏移量
    ty[b,best_n,gj,gi] = gy - gy.floor()

    tw[b,best_n,gj,gi] = torch.log(gw/anchors[best_n][:,0]+1e-16)
    th[b,best_n,gj,gi] = torch.log(gh/anchors[best_n][:,0]+1e-16)

    tcls[b,best_n,gj,gi,target_labels] =1
    class_mask[b,best_n,gj,gi] = (pred_cls[b,best_n,gj,gi].argmax(-1)==target_labels).float()
    iou_scores[b,best_n,gj,gi] = bbox_iou(pred_boxes[b,best_n,gj,gi],target_boxes,x1y1x2y2=False)
    tconf = obj_mask.float()
    return  iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf



def bbox_iou(pred_boxes,targets,x1y1x2y2=False):
    '''
    计算bbox的iou
    :param pred_boxes:预测框 XYHW // x1y1x2y2
    :param targets: GT框
    :return: iou值
    '''
    #取bbox左上角的点(x,y)
    b1_x1,b1_x2 = pred_boxes[:,0] - pred_boxes[:,2]/2,pred_boxes[:,0]+pred_boxes[:,2] / 2
    b2_x1,b2_x2 = targets[:,0] - targets[:,2]/2,targets[:,0]+targets[:,2] / 2
    b1_y1,b1_y2 = pred_boxes[:,1] + pred_boxes[:,3] / 2,pred_boxes[:,1] - pred_boxes[:,3] / 2
    b2_y1,b2_y2 = targets[:,1] + targets[:,3] / 2,targets[:,1] - targets[:,3] / 2

    inter_x1,inter_y1 = torch.max(b1_x1,b2_x1),torch.min(b1_y1,b2_y1)
    inter_x2,inter_y2 = torch.min(b1_x2,b2_x2),torch.max(b1_y2,b2_y2)

    inter_area = torch.clamp((inter_x2-inter_x1),min=0)*torch.clamp((inter_y1-inter_y2),min=0)
    b1_area = (b1_x2-b1_x1)*(b1_y1-b1_y2)
    b2_area = (b2_x2-b2_x1)*(b2_y1-b2_y2)

    comb_area = b1_area+b2_area-inter_area
    iou = inter_area / (comb_area+1e-16)

    return iou


test = torch.randn((2,3,4))
a = torch.tensor([[2,2,4,4],[2,2,4,4]]).float()
b = torch.tensor([[2.2,2,4,4],[2.2,2,4,4]]).float()

bbox_iou(a,b)











