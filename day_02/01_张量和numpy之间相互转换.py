'''
案例：
    演示张量 和 numpy之间如何相互转换， 以及如何从标量张量中 提取其中内容。

涉及到的函数：
    场景1： 张量 ——> numpy   nd数组
            张量对象.numpy()            共享内存
            张量对象.numpy().copy()     不共享内存， 链式编程写法

    场景2： numpy nd数组 ———> 张量
        from_numpy()                   共享内存
        torch.tensor(nd数组)           不共享内存

    场景3： 从标量张量中提取其内容
        标量张量.item()

需要掌握的函数：
    张量对象.numpy()，   torch.tensor(nd数组)， 标量张量.item()
'''

import torch
import numpy as np
from narwhals import from_numpy


# 1.定义函数， 演示：张量 ——> numpy
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}, type: {type(t1)}')

    n1 = t1.numpy()
    print(f'n1: {n1}, type: {type(n1)}')

    n1[0,0] = 100
    print(f'n1[0]: {n1[0]}, type: {type(n1[0])}, t1: {t1}, type: {type(t1)}')

    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t2}, type: {type(t2)}')

    n2 = t2.numpy().copy()
    print(f'n2: {n2}, type: {type(n2)}')

    n2[0,0] = 100
    print(f't2: {t2}, type: {type(t2)}')
    print(f'n2: {n2}, type: {type(n2)}')
# 2.定义函数， 演示：numpy ——> 张量
def dm02():
    t1 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}, type: {type(t1)}')

    n1 = torch.tensor(t1)
    print(f'n1: {n1}, type: {type(n1)}')

    n1[0, 0] = 255
    print(f'n1: {n1}, type: {type(n1)}'
          f't1: {t1}, type: {type(t1)}')

    t2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f't2: {t2}, type: {type(t2)}')

    n2 = torch.from_numpy(t2)       #共享内存
    print(f'n2: {n2}, type: {type(n2)}')

    n2[0 ,0] = 255
    print(f'n2: {n2}, type: {type(n2)}'
          f't2: {t2}, type: {type(t2)}')

# 3.定义函数， 演示：从标量张量中提取其内容
def dm03():
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')

    value = t1.item()
    print(f'value: {value}, type: {type(value)}')



# 测试函数
if __name__ == '__main__':
    # dm01()
    #dm02()
    dm03()