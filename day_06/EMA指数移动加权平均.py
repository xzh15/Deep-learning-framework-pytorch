'''
指数移动加权平均（Exponential Moving Average, EMA）
EMA 的核心逻辑：近重远轻      越新的数据对结果影响越大，越旧的数据影响越小，权重随时间指数衰减。
公式：EMA = α * 新数据 + (1 - α) * 上一步EMA
     EMA_t = α * x_t + (1-α) * EMA_{t-1}

todo:参数 α 是当前数据的权重， 是一个超参数，取值范围在 0 到 1 之间。α 的值越小，越看重历史数据，α 的值越大，越看重新数据

     连续5个batch的训练损失: x = [2.0, 1.8, 1.5, 1.7, 1.4]
     平滑系数: α = 0.3
     初始值: EMA_0 = x_0 = 2.0
     第1步（t=1，x=1.8）
     EMA_1 = 0.3 × 1.8 + 0.7 × 2.0 = 0.54 + 1.4 = 1.94
     第2步（t=2，x=1.5）
     EMA_2 = 0.3 × 1.5 + 0.7 × 1.94 = 0.45 + 1.358 = 1.808
     第3步（t=3，x=1.7）
     EMA_3 = 0.3 × 1.7 + 0.7 × 1.808 = 0.51 + 1.2656 = 1.7756
     第4步（t=4，x=1.4）
     EMA_4 = 0.3 × 1.4 + 0.7 × 1.7756 = 0.42 + 1.2429 = 1.6629
'''

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- 1. EMA 实现类 ----------------------
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ema = None  # 保存上一步的EMA值

    def update(self, x):
        if self.ema is None:
            # 第一步：初始值等于第一个观测值
            self.ema = x
        else:
            # 递推公式
            self.ema = self.alpha * x + (1 - self.alpha) * self.ema
        return self.ema


# ---------------------- 2. 生成模拟数据：带噪声的下降损失 ----------------------
np.random.seed(42)
steps = 100
# 真实趋势：指数下降
true_loss = 2.0 * np.exp(-0.03 * np.arange(steps))
# 加上随机噪声，模拟训练时的loss抖动
noisy_loss = true_loss + 0.15 * np.random.randn(steps)

# ---------------------- 3. 计算不同alpha的EMA ----------------------
# alpha=0.1：更平滑，滞后大
ema_smooth = EMA(alpha=0.1)
ema_smooth_values = [ema_smooth.update(x) for x in noisy_loss]

# alpha=0.5：更灵敏，噪声多
ema_fast = EMA(alpha=0.5)
ema_fast_values = [ema_fast.update(x) for x in noisy_loss]

# ---------------------- 4. 绘图对比 ----------------------
plt.figure(figsize=(12, 6))
plt.plot(noisy_loss, label='原始带噪声损失', color='gray', alpha=0.5, linewidth=1)
plt.plot(true_loss, label='真实趋势', color='black', linestyle='--', linewidth=1.5)
plt.plot(ema_smooth_values, label='EMA (α=0.1，更平滑)', color='#1f77b4', linewidth=2)
plt.plot(ema_fast_values, label='EMA (α=0.5，更灵敏)', color='#ff7f0e', linewidth=2)

plt.xlabel('训练步数', fontsize=11)
plt.ylabel('损失值', fontsize=11)
plt.title('不同平滑系数的EMA效果对比', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()
