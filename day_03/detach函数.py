'''
    detach()用来解决自动微分的弊端

    一个张量一旦设置了自动微分， 这个张量就不能直接转成numpy的ndarray对象了

    detach()函数可以用来创建一个张量的副本，这个副本不包含梯度信息，但是会共享内存

    核心: n1 = t1.detach().numpy()
'''

import torch

t1 = torch.tensor([10, 20], requires_grad = True, dtype = torch.float32)
print(f't1: {t1}, type: {type(t1)}')

n1 = t1.detach().numpy()
print(f'n1: {n1}, type: {type(n1)}')

t2 = t1.detach()
print(f't2: {t2}, type: {type(t2)}')
t2.data[0] = 100
print(f't1: {t1}, type: {type(t1)}')
print(f't2: {t2}, type: {type(t2)}')

print(f't1: {t1.requires_grad}')
print(f't2: {t2.requires_grad}')


