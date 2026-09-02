import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 准备一个简单的曲线拟合任务
x = torch.linspace(0, 1, 100).unsqueeze(1)
y = torch.sin(2 * 3.1416 * x) + 0.1 * torch.randn_like(x)

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 3. 换不同优化器测试
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

criterion = nn.MSELoss()
loss_history = []

# 4. 训练
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

# 5. 查看结果
print(f"最终损失: {loss_history[-1]:.4f}")

plt.plot(loss_history, label=type(optimizer).__name__)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("优化器收敛曲线对比")
plt.legend()
plt.show()
