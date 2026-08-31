'''
案例：
    演示二分类任务的交叉熵损失函数。

二分类交叉熵损失：BCEWithLogitsLoss()
    设计思路:

        Loss = - [y·log(预测值) + (1-y)·log(1-预测值))]

BCELoss 必须和 Sigmoid搭配
'''

import torch
import torch.nn as nn

# 定义二分类交叉熵损失（内置Sigmoid）
criterion = nn.BCEWithLogitsLoss()

# 模拟：batch_size=2个样本，二分类任务
# logits：模型原始输出，形状 [batch_size] = [2]
logits = torch.tensor([2.0, -1.5])

# target：真实标签，形状 [batch_size] = [2]
# 注意：二分类标签必须是 float 类型！这是和多分类的区别
target = torch.tensor([1.0, 0.0])

# 计算损失
loss = criterion(logits, target)
print(f"二分类交叉熵损失: {loss.item():.4f}")
# 输出：二分类交叉熵损失: 0.1643

