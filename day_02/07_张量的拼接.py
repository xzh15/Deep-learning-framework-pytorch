'''
案例：
    演示张量的拼接操作。

涉及到的函数：
    cat()       不会改变维度数， 在形状上拼接张量，除了拼接的的那个维度外，其他维度必须保持一致
    stack()     所有维度必须保持一致,   在维度上进行拼接
'''

import torch

t1 = torch.randint(1, 10, size = (2, 3))
print(f't1: {t1}, shape: {t1.shape}')

t2 = torch.randint(1, 10, size = (2,3))
print(f't2: {t2}, shape: {t2.shape}')

t3 = torch.cat([t1, t2], dim = 0)
print(f't3: {t3}, shape: {t3.shape}')

t4 = torch.cat([t1, t2], dim = 1)
print(f't4: {t4}, shape: {t4.shape}')\

t5 = torch.stack([t1, t2], dim = 0)
print(f't5: {t5}, shape: {t5.shape}')

t6 = torch.stack([t1, t2], dim = 1)
print(f't6: {t6}, shape: {t6.shape}')

t7 = torch.stack([t1, t2], dim = 2)
print(f't7: {t7}, shape: {t7.shape}')






