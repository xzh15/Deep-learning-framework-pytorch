import torch
# 1.定义变量， 记录：初始的权重 w (旧)
# 参1：初始值， 参2：是否自动微分（求导）， 参3：数据类型
w = torch.tensor(10, requires_grad= True, dtype = torch.float)

# 2.定义loss变量， 表示损失函数
loss = 2 * w ** 2

# 3. 打印梯度类型函数（了解）
print(f'梯度函数类型：{type(loss.grad_fn)}')
print(loss.sum())

# 4. 计算梯度， 梯度 = 损失函数的导数， 计算完毕后， 会记录到w.grad属性中
loss.sum().backward()       # 保证loss是一个标量

# 5. 带入权重更新公式： W新 = W旧 - 学习率 * 梯度
w.data = w.data - 0.01 * w.grad

# 6. 打印最终结果 吗
print(f'更新后的权重：{w}')


