'''
Smooth L1损失函数：结合MAE和MSE的优点，对小误差适用MAE，对大误差适用MSE。用于回归问题。
    设计思路：
        Loss = 1/n * Σ Smooth L1（ |y -预测值| ）
    优点：
        计算简单，结果容易解释。
    缺点：
        没有考虑预测值和真实值的差异大小，对大误差的惩罚较小。
    适用场景：
        适用于回归问题，特别是当数据中存在异常值时，可以使用Smooth L1损失函数来减轻异常值对模型的影响。

'''

import torch
import torch.nn as nn

# 定义Smooth L1损失，beta默认1.0
criterion = nn.SmoothL1Loss()

# 模拟：batch_size=3的回归任务
y_true = torch.tensor([80.0, 90.0, 70.0])
y_pred = torch.tensor([78.0, 95.0, 66.0])

# 计算损失
loss = criterion(y_pred, y_true)
print(f"Smooth L1损失值: {loss.item():.4f}")
# 输出：Smooth L1损失值: 3.1667

