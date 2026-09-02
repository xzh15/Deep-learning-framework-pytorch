import torch
import torch.nn as nn


def dm02():
    # 1. 创建隐藏层输出结果 (批量数据, batch_size=2, 特征数=4)
    t1 = torch.randint(0, 10, size=(2, 4)).float()
    # print(f't1: {t1}')

    # 2. 下一层：加权求和 + 批量归一化 + 激活函数
    # 2.1 创建全连接线性层: 输入4维，输出5维
    linear1 = nn.Linear(in_features=4, out_features=5)

    # 2.2 加权求和
    l1 = linear1(t1)
    print(f'l1(线性加权求和结果): {l1}')

    # 2.3 批量归一化 BN层
    # num_features: 传入特征的维度，这里是linear1输出的5
    bn1 = nn.BatchNorm1d(num_features=5)
    bn_out = bn1(l1)
    print(f'bn_out(BN批量归一化之后): {bn_out}')

    # 2.4 激活函数
    output = torch.relu(bn_out)
    print(f'output(relu激活后): {output}')


if __name__ == '__main__':
    dm02()
