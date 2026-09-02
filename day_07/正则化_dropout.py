import torch
import torch.nn as nn


def dm01():
    # 1. 创建隐藏层输出结果.
    t1 = torch.randint(0, 10, size=(1, 4)).float()
    # print(f't1: {t1}')    # t1: tensor([[0., 5., 6., 3.]])

    # 2. 进行下一层 加权求和 和 激活函数计算.
    # 2.1 创建全连接层(充当线性层)
    # 参1: 输入特征维度, 参2: 输出特征维度.
    linear1 = nn.Linear(in_features=4, out_features=5)

    # 2.2 加权求和.
    l1 = linear1(t1)
    print(f'l1: {l1}')

    # 2.3 激活函数.
    output = torch.relu(l1)
    print(f'output: {output}')

    # 3. 对激活值进行随机失活dropout处理 → 只有训练阶段有, 测试阶段没有.
    dropout = nn.Dropout(p=0.5)  # 每个神经元都有40%的概率被 kill.
    # 具体的 随机失活动作.
    d1 = dropout(output)
    print(f'd1(随机失活后的数据): {d1}')


if __name__ == '__main__':
    dm01()
