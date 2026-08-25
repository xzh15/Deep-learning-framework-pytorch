'''
案例：
    演示张量的基本运算

涉及到的函数：
    add(), sub(), mul(), div(), neg()
    add_(), sub_(), mul_(), div_(), neg_()      #修改原数据

需要记忆的：
    + — * /
'''

import torch
import numpy as np

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f't1: {t1}')

t2 = t1.neg()
print(f't2: {t2}')

t2 = t1 + 10
print(f't2: {t2}')

t2 = t1 - 10
print(f't2: {t2}')