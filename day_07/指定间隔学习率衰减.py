import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 准备数据：带噪声的正弦曲线拟合 ----------------------
x = torch.linspace(0, 2, 200).unsqueeze(1)
y = torch.sin(2 * 3.1416 * x) + 0.1 * torch.randn_like(x)

# ---------------------- 2. 定义模型、优化器、学习率调度 ----------------------
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 优化器：带动量的SGD
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 指定间隔学习率衰减
# milestones=[30, 70]：在第30、70个epoch分别触发一次衰减
# gamma=0.1：每次衰减为原来的10%
scheduler = MultiStepLR(optimizer, milestones=[30, 70], gamma=0.1)

criterion = nn.MSELoss()

# ---------------------- 3. 训练循环 ----------------------
loss_history = []
lr_history = []

for epoch in range(100):
    # 前向传播 + 计算损失
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # 反向传播 + 更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每个epoch结束，调用调度器更新学习率
    scheduler.step()

    # 记录数据
    current_lr = optimizer.param_groups[0]['lr']
    loss_history.append(loss.item())
    lr_history.append(current_lr)

    # 关键节点打印
    if (epoch + 1) in [1, 30, 31, 70, 71, 100]:
        print(f"Epoch {epoch + 1:3d} | 损失: {loss.item():.4f} | 当前学习率: {current_lr:.6f}")

# ---------------------- 4. 可视化结果 ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 损失曲线
ax1.plot(loss_history, label='训练损失', color='#1f77b4')
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('指定间隔学习率衰减的训练损失曲线', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 学习率变化曲线
ax2.plot(lr_history, label='学习率', color='#ff7f0e', linewidth=2)
ax2.set_xlabel('训练轮次 Epoch', fontsize=11)
ax2.set_ylabel('学习率（对数坐标）', fontsize=11)
ax2.set_title('学习率阶梯式变化（里程碑：30、70）', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
