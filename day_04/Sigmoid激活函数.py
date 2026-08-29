'''
Sigmoid 激活函数：
主要应用于 二分类的输出层，且适用于 浅层神经网络 (不超过 5 层)。
数据在 [-6, 6] 之间有效果，在 [-3, 3] 之间效果明显，会将数据值映射到: [0, 1]
求导后范围在 [0, 0.25]
'''

import torch
import matplotlib.pyplot as plt

# 配置中文与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建 1行2列 的子图画布
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# ---------------------- 左图：Sigmoid 函数曲线 ----------------------
# 2. 生成 [-20, 20] 区间内的1000个均匀采样点
x = torch.linspace(-20, end=20, steps=1000)

# 3. 计算Sigmoid函数值
y_sigmoid = torch.sigmoid(x)

# 4. 绘制函数图像
axes[0].plot(x.numpy(), y_sigmoid.numpy(), linewidth=2, color='#1f77b4')
axes[0].set_title('Sigmoid 激活函数', fontsize=12)
axes[0].set_xlabel('x', fontsize=10)
axes[0].set_ylabel('σ(x)', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# ---------------------- 右图：Sigmoid 导数曲线 ----------------------
# 5. 生成开启梯度追踪的输入张量
x_grad = torch.linspace(-20, end=20, steps=1000, requires_grad=True)

# 6. 前向传播计算Sigmoid
y_grad = torch.sigmoid(x_grad)

# 7. 反向传播计算导数：输出为向量，先求和转为标量再反向传播
y_grad.sum().backward()

# 8. 提取梯度值（即导数值）
dy_dx = x_grad.grad

# 9. 绘制导数图像
axes[1].plot(x_grad.detach().numpy(), dy_dx.numpy(), linewidth=2, color='#ff7f0e')
axes[1].set_title('Sigmoid 导数函数', fontsize=12)
axes[1].set_xlabel('x', fontsize=10)
axes[1].set_ylabel("σ'(x)", fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 自动调整子图间距
plt.tight_layout()
plt.show()
