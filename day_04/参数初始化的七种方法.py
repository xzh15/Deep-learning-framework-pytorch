'''
案例：
    演示参数初始化的七种方式

参数初始化的目的：
    1.防止梯度消失 或者 爆炸
    2.提高收敛速度
    3.打破对称性

参数初始化的方法：
    无法打破对称性的：
        全 0，全 1，固定值

    可以打破对称性的：
        随机初始化，正态分布初始化，kaiming 初始化，xavier 初始化

总结：
    1. 记忆 kaiming 初始化，xavier 初始化，全 0 初始化。
    2. 关于初始化的选择上：
        激活函数 ReLU 及其系列：优先用 kaiming
        激活函数非 ReLU：优先用 xavier
        如果是浅层网络：可以考虑使用 随机初始化
'''
import torch
import torch.nn as nn
torch.manual_seed(24)

# 1.均匀分布随机初始化
    # 从0-1均匀分布产生参数
def dm01():
    linear = nn.Linear(5, 3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

# 2.固定初始化
def dm02():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 0.5)
    nn.init.constant_(linear.bias, 0.5)
    print(linear.weight.data)
    print(linear.bias.data)

# 3.全0初始化
def dm03():
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

# 4.全1初始化
def dm04():
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

# 5.正态分布随机初始化
def dm05():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

# 6.kaiming初始化
def dm06():
    # linear = nn.Linear(5, 3)
    # nn.init.kaiming_normal_(linear.weight)
    # print(linear.weight.data)
    # print(linear.bias.data)

    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight)

    print(linear.weight.data)
    print(linear.bias.data)

# 7.xavier 初始化
def dm07():
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)

    print(linear.weight.data)
    print(linear.bias.data)

# 测试
if __name__ == '__main__':
    # dm01()
    dm06()
