"""
StepLR 等间隔阶梯式学习率衰减
【功能】
    每隔固定数量的训练周期（epoch），按固定比例对 全局学习率 进行阶梯式衰减，整体呈阶梯状下降。
    核心作用是平衡训练速度与最终收敛精度：
    - 训练前期：保持较大学习率，快速逼近最优解区域，大幅加快收敛速度
    - 训练后期：自动降低学习率，减小参数更新幅度，进行精细微调，提升收敛精度
    解决了固定学习率「过大易震荡发散、过小收敛极慢」的两难问题。

【核心公式】
    lr_t = lr_initial * gamma ^ floor(t / step_size)
    参数说明：
    - lr_initial : 优化器设定的初始基础学习率
    - gamma     : 衰减系数（取值 0 < gamma < 1），每次衰减后学习率变为原来的 gamma 倍
    - step_size : 衰减间隔（单位：epoch），每经过 step_size 个完整训练周期触发一次衰减
    - t         : 当前训练的 epoch 序号
    - floor()   : 向下取整函数

【具体运用】
    1. 使用方式：与优化器绑定创建，每个 epoch 训练完成后调用 scheduler.step() 更新学习率
    2. 典型参数取值：
        - step_size：常取 20 / 30 / 50 / 100，根据总训练轮次灵活调整
        - gamma：常用 0.1（衰减为原来1/10）、0.5（衰减为原来1/2）、0.9（小幅平滑衰减）
    3. 特性与注意：
        - 属于「全局级学习率调整」，所有参数的学习率同比例缩放；
          与 Adam/RMSProp 等「参数级自适应学习率」不冲突，可叠加配合使用
        - 衰减为阶梯式突变，而非平滑过渡
    4. 适用场景：绝大多数基础训练任务通用，尤其适合快速原型验证、
        对调度精度要求不高的场景，是学习率调度的基础选型。
"""
from torch.optim.lr_scheduler import StepLR

# 示例：绑定优化器使用
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 每轮训练结束后调用：scheduler.step()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


x = torch.linspace(0, 2, 200).unsqueeze(1)
y = torch.sin(2 * 3.1416 * x) + 0.1 * torch.randn_like(x)

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum= 0.9)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

criterion = nn.MSELoss()

loss_history = []
lr_history = []

for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    loss_history.append(loss.item())
    lr_history.append(current_lr)

    if(epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], 损失Loss: {loss.item():.4f}, 当前学习率: {current_lr:.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 损失曲线
ax1.plot(loss_history, label='训练损失', color='#1f77b4')
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('等间隔学习率衰减的训练损失曲线', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 学习率变化曲线
ax2.plot(lr_history, label='学习率', color='#ff7f0e', linewidth=2)
ax2.set_xlabel('训练轮次 Epoch', fontsize=11)
ax2.set_ylabel('学习率（对数坐标）', fontsize=11)
ax2.set_title('学习率阶梯式变化', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # 对数坐标，方便看阶梯

plt.tight_layout()
plt.show()