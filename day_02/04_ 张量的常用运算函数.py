'''
案例：
    演示张量常用的运算函数。

涉及到的函数：
   sum(), max(), min(), mean()
   pow(), sqrt(), exp(), log(), log2(), log10()

需要掌握的函数：
    sum(), max(), min(), mean()
'''

import torch

t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]], dtype=torch.float)
print(f't1: {t1}')

print(t1.sum(dim=0))        #按 列 求和
print(t1.sum(dim=1))        #按 行 求和
print(t1.sum())             #整 体 求和
print('-' * 30)

print(t1.max(dim=0))        #按 列 求最大值
print(t1.max(dim=1))        #按 行 求最大值
print(t1.max())             #整 体 求最大值
print('-' * 30)

print(t1.mean(dim=0))        #按 列 求平均值
print(t1.mean(dim=1))        #按 行 求平均值
print(t1.mean())             #整 体 求平均值
print('-' * 30)

print(t1.pow(2))        #每个数的平方
print(t1.pow(3))        #每个数的立方
print(t1 ** 2)
print('-' * 30)

print(t1.sqrt_())
print('-' * 30)

print(t1.exp())
print('-' * 30)

print(t1.log())
print(t1.log2())
print(t1.log10())
