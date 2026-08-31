'''
MSE损失函数：Mean Squared Error，均方误差。用于回归问题。
    设计思路：
        Loss = 1/n * Σ(y - 预测值)^2
    优点：
        计算简单，结果容易解释。
    缺点：
        对大误差的惩罚较大。
'''

import torch
import torch.nn as nn

# 定义MSE损失（L2损失）
criterion = nn.MSELoss()

# 模拟：batch_size=3的回归任务
y_true = torch.tensor([80.0, 90.0, 70.0])
y_pred = torch.tensor([78.0, 95.0, 66.0])

# 计算损失
loss = criterion(y_pred, y_true)
print(f"MSE损失值: {loss.item():.4f}")
# 输出：MSE损失值: 15.0000
