"""
自适应学习率：AdaGrad

公式：
累计平方梯度：
St = St-1 + Gt * Gt

解释：
St:        累计平方梯度
St-1:      历史累计平方梯度.
Gt:        本次的梯度.

学习率：
学习率 = 学习率 / (sqrt(St) + 小常数)

解释：
小常数: 1e-10, 目的: 防止分母变为0

梯度下降公式：
W新 = W旧 - 调整后的学习率 * Gt

缺点:
可能会导致学习率过早，过量的降低，导致模型后期学习率太小，较难找到最优解.
"""

import torch
import torch.optim as optim


def dm02_adagrad():
    # 1. 初始化权重参数。
    w = torch.tensor(data=[1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器(函数对象)
    # 思路1: 基于SGD(随机梯度下降), 加入参数 momentum, 就是动量法。
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数。
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑本次梯度。

    # 思路2: 基于AdaGrad(自适应学习率)。
    optimizer = optim.Adagrad(params=[w], lr=0.01)

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
    dm02_adagrad()

