



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













Feature: