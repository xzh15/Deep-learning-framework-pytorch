'''

MAE损失函数：Mean Absolute Error，平均绝对误差。用于回归问题。
    设计思路：
        Loss = 1/n * Σ|y -预测值|
    优点：
        计算简单，结果容易解释。
    缺点：
        没有考虑预测值和真实值的差异大小，对大误差的惩罚较小。

'''

import torch
import torch.nn as nn

# 定义MAE损失（L1损失）
criterion = nn.L1Loss()

# 模拟：batch_size=3的回归任务
# 真实值（连续数值），形状 [batch_size]
y_true = torch.tensor([80.0, 90.0, 70.0])
# 模型预测值，形状和真实值完全一致
y_pred = torch.tensor([78.0, 95.0, 66.0])

# 计算损失
loss = criterion(y_pred, y_true)
print(f"MAE损失值: {loss.item():.4f}")
# 输出：MAE损失值: 3.6667
