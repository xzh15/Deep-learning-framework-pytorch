'''
案例：
    演示张量的索引操作
分类：
    简单行列索引
    列表索引
    范围索引
    布尔索引
    多维索引
掌握：
    简单行列索引， 范围索引， 多维索引(重点理解)

'''

import torch

torch.manual_seed(24)

t1 = torch.randint(0, 10, (5, 5))
print(f't1: {t1}')
print('-' * 30)

# 简单行列索引
print(t1[1 , :])
print(t1[: , 2])

# 列表索引 前一个[]里是行，后一个[]是列，所以匹配的是（1，2）和（3，4）
print(t1[[1, 3], [2, 4]])
# 获取第0， 1行的 1， 2列，四个元素
print(t1[[[0], [1]], [1, 2]])
print('-' * 30)

# 范围索引
print(t1[:3, :2])
print(t1[1:, :2])
# 所有奇数行，偶数列
print(t1[1::2, ::2])
print('-' * 30)

# 布尔索引
print(t1[torch.tensor([True, False, False, False, False]),:])

# 第2列大于5的行数据
print(t1[t1[:, 2] > 5])
print(t1[:, t1[1, :] > 5])
print(t1[1, t1[1, :] > 5])

# 多维索引
t2 = torch.randint(1, 10, (2, 3, 4))
print(f't2: {t2}')
print('-' * 30)
print(t2[0, :, :])
print('-' * 30)
print(t2[:, 0, :])
print('-' * 30)
print(t2[:, :, 0])
