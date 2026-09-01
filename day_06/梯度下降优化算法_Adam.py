"""
自适应矩估计：Adam(Adaptive Moment Estimation)

思路:
即优化学习率，又优化梯度。

公式:
一阶矩：算均值。
Mt = β1 * Mt-1 + (1 - β1) * Gt      充当: 梯度
St = β2 * St-1 + (1 - β2) * Gt * Gt  充当: 学习率

二阶矩：梯度的方差。
Mt^ = Mt / (1 - β1 ^ t)
St^ = St / (1 - β2 ^ t)

权重更新公式:
W新 = W旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^

大白话翻译:
Adam = RMSProp + Momentum
"""
import torch
import torch.optim as optim

def dm04_adam():
    # 1. 初始化权重参数。
    w = torch.tensor(data=[1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器(函数对象)

    optimizer = optim.Adam(params=[w], lr=0.01)

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
    dm04_adam()