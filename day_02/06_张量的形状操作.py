'''
案例：
    演示张量的形状操作。

涉及到的函数：
    reshape()           在不改变张量内容的前提下，对其形状做改变
    unsqueeze()         在指定的轴上增加一个（1）维度
    squeeze()           删除所有为 1 的维度， 等价于：降维
    transpose()         一次只能交换两个维度
    permute()           一次可以同时交换多个维度
    view()              只能修改连续的张量的形状，连续的张量 = 内存中存储顺序 和 张量中显示的顺序相同
    contiguous()        把不连续的张量 --> 连续的张量， 即：基于张量中显示的顺序，修改为内存中的存储顺序
    is_contiguous()     判断张量是否是连续的

需要掌握的函数：
    reshape(), unsqueeze(), permute(), view()
'''

import torch
torch.manual_seed(24)

# reshape(),数据元素内容不能改变
def dm01():
    t1 = torch.randint(1, 10, size = (2, 3))
    print(f't1: {t1}, shape: {t1.shape}, row: {t1.shape[0]}, columns: {t1.shape[1]}, {t1.shape[-1]}')

    t2 = t1.reshape(3, 2)
    print(f't2: {t2}, shape: {t2.shape}, row: {t2.shape[0]}, columns: {t2.shape[1]}, {t2.shape[-1]}')

    t3 = t1.reshape(1, 6)
    print(f't3: {t3}, shape: {t3.shape}, row: {t3.shape[0]}, columns: {t3.shape[1]}, {t3.shape[-1]}')

# unsqueeze()   squeeze()
def dm02():
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1: {t1}, shape: {t1.shape}')

    t2 = t1.unsqueeze(0)
    print(f't2: {t2}, shape: {t2.shape}')      # (1, 2, 3)

    t3 = t1.unsqueeze(1)
    print(f't3: {t3}, shape: {t3.shape}')      # (2, 1, 3)

    t4 = t1.unsqueeze(2)
    print(f't4: {t4}, shape: {t4.shape}')      # (2, 3, 1)

    # t5 = t1.unsqueeze(3)                     报错
    # print(f't5: {t5}, shape: {t5.shape}')    # (2, 3, *, 1)

    t6 = torch.randint(1, 10, size = (2, 1, 3, 1, 1))
    print(f't6: {t6}, shape: {t6.shape}')

    t7 = t6.squeeze()
    print(f't7: {t7}, shape: {t7.shape}')

# transpose()   permute()
def dm03():
    t1 = torch.randint(1, 10, size = (2, 3, 4))
    print(f't1: {t1}, shape: {t1.shape}')

    t2 = t1.transpose(0, -1)
    print(f't2: {t2}, shape: {t2.shape}')  # (4, 3, 2)

    t3 = t1.permute(2, 0 ,1)
    print(f't3: {t3}, shape: {t3.shape}')   # (4 ,2, 3)


# view()    contiguous()    is_contiguous()
def dm04():
    t1 = torch.randint(1, 10, size = (2, 3))
    print(t1.is_contiguous())
    print(f't1: {t1}, shape: {t1.shape}')

    t2 = t1.view(3, 2)
    print(f't2: {t2}, shape: {t2.shape}')
    print(t2.is_contiguous())

    # 通过transpose()交换维度 --> 交换之后不连续了
    t3 = t1.transpose(0, 1)
    print(f't3: {t3}, shape: {t3.shape}')
    print(t3.is_contiguous())

    t4 = t3.contiguous().view(2, 3)
    print(f't4: {t4}, shape: {t4.shape}')
    print(t4.is_contiguous())

# 测试函数
if __name__ == '__main__':
    # dm01()
    dm02()
    # dm03()
    # dm04()



