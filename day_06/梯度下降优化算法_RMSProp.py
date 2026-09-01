"""
自适应学习率：RMSProp → 可以看做是 对AdaGrad做的优化，加入 调和权重系数。

公式：
指数加权平均 累计历史平方梯度：
St = β * St-1 + (1 - β) * Gt * Gt

解释：
St:        累计平方梯度
St-1:      历史累计平方梯度.
Gt:        本次的梯度.
β:         调和权重系数.

学习率：
学习率 = 学习率 / (sqrt(St) + 小常数)

解释：
小常数: 1e-10, 目的: 防止分母变为0

梯度下降公式：
W新 = W旧 - 调整后的学习率 * Gt

优点:
RMSProp通过引入 衰减系数β，控制历史梯度 对 历史梯度信息获取的多少。
"""
import torch
import torch.optim as optim

def dm03_rmsprop():
    # 1. 初始化权重参数。
    w = torch.tensor(data=[1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器(函数对象)

    optimizer = optim.RMSprop(params=[w], lr=0.01)

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
    dm03_rmsprop()
