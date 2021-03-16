# WSOL 弱监督目标定位源码解读

采用的代码是[Evaluating Weakly Supervised Object Localization Methods Right (CVPR 2020)](https://github.com/clovaai/wsolevaluation)，本文主要对该代码进行解析学习。

## 代码结构

![image-20210316190737600](image-20210316190737600.png)

如上图，

- dasaset文件夹就是用来存放数据集的，将下载好的数据放在这个文件中，具体存放的格式可以在readme文件中找到；
- metadata用来存放数据集的元数据，包括一些标注信息；
- test_data中数据便于用户进行代码的调试；
- wsol文件中存放的就是作者复现的弱监督目标定位的常用方法以及网络架构；外面的文件中config是配置文件，在这里进行各种参数的配置；
- data_loders文件创建data_loader类用来对数据集中的图片进行预处理以及获取图片的元数据等；
- evaluation文件是用来存放对结果进行评价的函数(计算cam，计算IOU等)；
- evaluation_test对evaluation中的函数进行测试；
- inference是用来推断的文件；
- main文件是主文件，主函数在这里执行；
- util中存放一些会用到的函数

下面就从主函数进行切入对所有用到的函数进行分析与介绍：

```python
def main():
    trainer = Trainer()

    print("===========================================================")
    print("Start epoch 0 ...")
    trainer.evaluate(epoch=0, split='val')
    trainer.print_performances()
    trainer.report(epoch=0, split='val')
    trainer.save_checkpoint(epoch=0, split='val')
    print("Epoch 0 done.")

    for epoch in range(trainer.args.epochs):
        print("===========================================================")
        print("Start epoch {} ...".format(epoch + 1))
        trainer.adjust_learning_rate(epoch + 1)
        train_performance = trainer.train(split='train')
        trainer.report_train(train_performance, epoch + 1, split='train')
        trainer.evaluate(epoch + 1, split='val')
        trainer.print_performances()
        trainer.report(epoch + 1, split='val')
        trainer.save_checkpoint(epoch + 1, split='val')
        print("Epoch {} done.".format(epoch + 1))

    print("===========================================================")
    print("Final epoch evaluation on test set ...")

    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    # trainer.load_checkpoint('best')
    trainer.evaluate(trainer.args.epochs, split='test')
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()
```

主函数非常简洁：

- 首先创建了一个名为 trainer 的对象，然后就直接调用其中的函数进行了一次评价的过程，评价之后输出各项指标表现，然后通过调用report函数对结果进行记录，最后对model进行存储；
- 在进行一次评价之后就开始对模型进行训练，整个训练的次数由config文件中epoch来指定，训练的过程中需要对学习率进行调整，对训练后模型的表现进行评价以及保存model
- 训练过后就需要在测试集上查看表现并对模型表现结果进行输出

## Trainer()类的构建

在设置好数据集后，我们对该模型进行debug调试。

首先进入到主函数中，第一行就是这个类，进入到这个类中对其进行查看。

![image-20210316201141649](image-20210316201141649.png)

```python

class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = self._set_optimizer() #设置优化方式
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    def _set_performance_meters(self):
    def _set_model(self):
    def _set_optimizer(self):
    def _wsol_training(self, images, target):
    def train(self, split):
    def print_performances(self):
    def save_performances(self):
    def _compute_accuracy(self, loader):
    def evaluate(self, epoch, split):
    def _torch_save_model(self, filename, epoch):
    def save_checkpoint(self, epoch, split):    
    def report_train(self, train_performance, epoch, split='train'):
    def report(self, epoch, split):
    def adjust_learning_rate(self, epoch):
    def load_checkpoint(self, checkpoint_type):
```

Trainer这个类中含有的函数较多，我们一个个慢慢解读：

首先创建了一些变量：

```python
  _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }
```

这些带有单个下划线的变量仅供内部使用

之后进行了类中变量的初始化：

```python
  def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.model = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer = self._set_optimizer() #设置优化方式
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)
```

其中 `self.args` 从config文件中调用函数得到所有的参数，该变量是一个`argparse.Namespace`对象，主要用于对参数存储，关于argparse的详细介绍可以查看[`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) --- 命令行选项、参数和子命令解析器。

![image-20210316202650381](image-20210316202650381.png)

然后分别设置了一个随机数，对评价指标进行设置，对模型进行初始化



Images: 输入图片维度 $batch\_size*3*224*224$

Taget:输入图片的标签 $batch\_size$



在forward的过程中：

Feature:提取特征部分 $batch\_size*512*28*28$

Feature_map:经过分类后$batch\_size*classes(200)*28*28$

Logits:输出分类$batch\_size*classes(200)$



Attention:（对应于正确类别的feature_map$batch\_size*class(1)*28*28$

pos：找到大于一定值的部分，代表feature中受到关注的部分 $batch\_size*class(1)*28*28$

```
pos = torch.ge(attention, drop_threshold)
```

数据形式是false和true

## 参数配置











Feature: