'''
梯度下降是结合本次损失函数的导数(作为梯度)、基于学习率来更新权重的。
公式:
    W新 = W旧 - 学习率 * (本次的)梯度

存在的问题:
    1. 遇到平缓区域, 梯度下降(权重更新)可能会慢。
    2. 可能会遇到鞍点(梯度为0)
    3. 可能会遇到局部最小值。

解决思路:
    从上述的学习率或者梯度入手进行优化, 于是有了:
    动量法 Momentum, 自适应学习率 AdaGrad, RMSProp, 综合衡量: Adam

 动量法公式:
 St = β * St-1 + (1 - β) * Gt
 解释:
 St:   本次的指数移动加权平均结果。
 β:    调节权重系数, 越大, 数据越平缓, 历史指数移动加权平均比重越大, 本次梯度权重越小。
 St-1: 历史的指数移动加权平均结果。
 Gt:   本次计算出的梯度(不考虑历史梯度)。

 加入动量法后的梯度更新公式:
 W新 = W旧 - 学习率 * St
'''

import torch
import torch.optim as optim


def dm01_momentum():
    # 1. 初始化权重参数。
    w = torch.tensor(data=[1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器(函数对象) -> 基于SGD(随机梯度下降), 加入参数 momentum, 就是动量法。
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数。
    # 细节: momentum=0(默认), 只考虑本次梯度。
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)

    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

    # 5. 重复上述的步骤, 第2次更新权重参数。
    # 5.1 定义损失函数。
    criterion = ((w ** 2) / 2.0)

    # 5.2 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()

    # 5.3 打印结果。
    print(f'w: {w}, w.grad: {w.grad}')


if __name__ == '__main__':
    dm01_momentum()
