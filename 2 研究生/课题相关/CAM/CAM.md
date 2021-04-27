# CAM

## Axiomatic Attribution for Deep Networks

定义1：贡献定义

假设深度网络学习到的函数为$F$其输出的取值范围是$[0,1]$，假设输入为$x=(x_1,...,x_n)$，经过网络后输出为$F(x)$，假设一个基准输入为$x’=(x_1’,...,x_n’)$，其经过网络的输出为$F(x’)$，则我们可以通过一个函数来衡量输入$x$相对于基准$x’$的贡献，意即我们的输入对于网络的贡献，设该表示为$A_F(x,x’)=(a_1,a_2,...,a_n)$



公理1：敏感性（Sensitivuty）

敏感性指的是当输入有一个feature不同且输出预测值不同的时候那么此feature应该有一个不为0的贡献。

![image-20210426192629792](../../%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CV/CAM/image-20210426192629792.png)

- 梯度方式违反敏感性原则：由于RuLU函数的性质，尽管输入与baseline不同其梯度值也可能为0，与baseline相比没有变化。RuLU函数在x>0的时候梯度值都是1.

公理2：输入不变性（Implementation Invariance）

当网络实现不同但是相同的输入都有相同的输出（功能上相同）时其对应的贡献应该相同。

![image-20210426195627490](../../%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CV/CAM/image-20210426195627490.png)

本文提出的积分梯度法：

![image-20210426210054000](../../%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CV/CAM/image-20210426210054000.png)

实际实现的方式：

![image-20210426210941789](../../%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/CV/CAM/image-20210426210941789.png)

```python
# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    delta_X = (pre_processing(inputs, cuda) - pre_processing(baseline, cuda)).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0))
    integrated_grad = delta_X * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                                baseline=255.0 *np.random.random(inputs.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
```

代码地址：https://github.com/TianhongDai/integrated-gradient-pytorch

参考资料：

https://zhuanlan.zhihu.com/p/148105536

https://distill.pub/2020/attribution-baselines/

