'''
Sigmoid 激活函数：
主要应用于 二分类的输出层，且适用于 浅层神经网络 (不超过 5 层)。
数据在 [-6, 6] 之间有效果，在 [-3, 3] 之间效果明显，会将数据值映射到: [0, 1]
求导后范围在 [0, 0.25]

Tanh：
主要应用于 隐藏层，且适用于 浅层神经网络 (不超过 5 层)。
数据在 [-3, 3] 之间有效果，在 [-1, 1] 之间效果明显，会将数据值映射到: [-1, 1]
求导后范围在 [0, 1]，较之于 Sigmoid，收敛速度快。

ReLU：
计算公式为：max (0, x)，计算量相对较小，训练成本低。多应用于隐藏层，且适合深层神经网络。
求导后，值要么是 0，要么是 1，较之于 Tanh，收敛速度更快。
默认情况下 ReLU 只考虑正样本，可以使用 LeakyReLU、PReLU 来考虑正负样本。

Softmax：
将多分类的结果以概率的形式展示，且概率和相加为 1，最终选取概率值最大的分类作为最终结果。
'''

import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义绘图区间和采样点数
x_range = (-10, 10)
n_points = 1000

# 创建 2行4列 的子图：第一行函数曲线，第二行导数曲线
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# 激活函数配置：名称、PyTorch实现、绘制颜色
activations = [
    ("Sigmoid", torch.sigmoid, '#1f77b4'),
    ("Tanh", torch.tanh, '#ff7f0e'),
    ("ReLU", torch.relu, '#2ca02c'),
    ("Softmax", lambda x: torch.softmax(x, dim=0), '#d62728')
]

for col_idx, (name, func, color) in enumerate(activations):
    # ---------- 绘制函数曲线（第一行） ----------
    x = torch.linspace(*x_range, steps=n_points)
    y = func(x)

    axes[0, col_idx].plot(x.numpy(), y.numpy(), linewidth=2, color=color)
    axes[0, col_idx].set_title(f'{name} 激活函数', fontsize=12)
    axes[0, col_idx].grid(True, alpha=0.3)
    axes[0, col_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, col_idx].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # ---------- 绘制导数曲线（第二行） ----------
    x_grad = torch.linspace(*x_range, steps=n_points, requires_grad=True)
    y_grad = func(x_grad)
    y_grad.sum().backward()
    dy_dx = x_grad.grad

    axes[1, col_idx].plot(x_grad.detach().numpy(), dy_dx.numpy(), linewidth=2, color=color)
    axes[1, col_idx].set_title(f'{name} 导数函数', fontsize=12)
    axes[1, col_idx].grid(True, alpha=0.3)
    axes[1, col_idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, col_idx].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# 自动调整子图间距，避免标题重叠
plt.tight_layout()
plt.show()
