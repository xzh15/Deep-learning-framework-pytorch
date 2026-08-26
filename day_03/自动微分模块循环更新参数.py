'''
案例：
    演示自动微分模块的使用。循环更新参数
需求：
    求 y = x ** 2 + 20 的极小值点，并打印 y 是最小值时 w 的值
解题步骤：
    1. 定义点 x = 10, requires_grad = True  dtype = torch.float32
    2. 定义函数 y = x ** 2 + 20
    3. 利用梯度下降法 循环迭代1000 求最优解
    3.1 正向计算（前向传播）
    3.2 梯度清零 x.grad.zero_()
    3.3 反向传播
    3.4 参数更新 x.data = x.data - 0.01 * x.grad
'''

import torch

# 1. 定义点 x = 10, requires_grad = True  dtype = torch.float32
w = torch.tensor(10, requires_grad= True, dtype = torch.float32)


# 2. 定义函数 y = x ** 2 + 20
loss = w ** 2 + 20

# 3. 利用梯度下降法 循环迭代100 求最优解
print(f'开始权重初始值： {w}, (0.01 * w.grad)')
for i in range(1,101):
    # 3.1 正向计算（前向传播）
    loss = w ** 2 + 20
    # 3.2 梯度清零 x.grad.zero_()
    if w.grad is not None:
        w.grad.zero_()
    # 3.3 反向传播
    loss.sum().backward()
    # 3.4 参数更新 x.data = x.data - 0.01 * x.grad
    w.data = w.data - 0.01 * w.grad
    print(f'第{i}次,权重初始值：{w}, (0.01 * w.grad):{0.01 * w.grad:.5f}, loss: {loss:.5f}')
print(f'最终结果权重:{w},  梯度：{w.grad}')