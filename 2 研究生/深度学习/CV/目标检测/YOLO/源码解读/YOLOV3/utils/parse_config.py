




#读取模型的配置文件
def parse_model_config(path):
    file = open(path,"r")
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        # 当检测到一个新的模块的时候将其加入一个字典
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


if __name__ == '__main__':
    path = "/Users/lianxing/Desktop/server/PyTorch-YOLOv3.nosync/config/yolov3.cfg"
    file = open(path,"r")
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        #当检测到一个新的模块的时候将其加入一个字典
        if line.startswith("["):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key,value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    a=2








