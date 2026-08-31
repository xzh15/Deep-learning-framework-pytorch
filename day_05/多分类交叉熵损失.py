'''
案例：
    演示多分类任务的交叉熵损失函数。
损失函数介绍：
    概述：
        损失函数也叫成本函数、目标函数、代价函数、误差函数，就是用来衡量模型好坏（模型拟合情况）的。
分类：
    分类问题：
         多分类交叉熵损失：CrossEntropyLoss
         二分类交叉熵损失：BCELoss
    回归问题：
        MAE：Mean Absolute Error，平均绝对误差。
        MSE：Mean Squared Error，均方误差。
        Smooth L1：结合上述两个的特点做的升级、优化。

多分类交叉熵损失：CrossEntropyLoss
    设计思路：Loss = - Σ y·log(S(f(x)))
            | 符号 | 含义 |
            | x | 样本 |
            | f(x) | 加权求和 |
            | S(f(x)) | 处理后的概率 |
            | y | 样本 x 属于某一个类别的真实概率 |
'''

import torch
import torch.nn as nn

# 模拟：3分类任务，batch_size=2个样本
# logits：模型最后一层的原始输出，形状 [batch_size, 类别数] = [2, 3]
logits = torch.tensor([
    [2.0, 1.0, 0.1],  # 第1个样本：模型认为第0类概率最高
    [0.5, 2.5, 1.0]   # 第2个样本：模型认为第1类概率最高
])

# target：真实类别索引，形状 [batch_size] = [2]
# 第1个样本真实类别是0，第2个样本真实类别是1
target = torch.tensor([0, 1], dtype=torch.long)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(logits, target)
print(f"交叉熵损失值: {loss.item():.4f}")



