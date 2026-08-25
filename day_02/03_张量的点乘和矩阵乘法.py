'''
案例：
    演示张量 的点乘和矩阵乘法。

点乘：
    要求两个张量的维度保持一致，对应元素直接相乘
    t1 * t2
    t1.mul(t2)
矩阵乘法：
    要求满足矩阵乘法，A列 = B行
    t1 @ t2
    t1.matmul(t2)
'''

import torch

# 点乘
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}')

    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't2: {t2}')

    t3 = t1 * t2
    print(f't3: {t3}')

def dm02():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}')

    t2 = torch.tensor([[1, 2], [4, 5], [7, 8]])
    print(f't2: {t2}')

    t3 = t1 @ t2
    print(f't3: {t3}')


# 测试
if __name__ == '__main__':
    #dm01()
    dm02()