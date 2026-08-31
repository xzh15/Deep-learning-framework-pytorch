'''
设计要求：
    1. 第 1 个隐藏层：权重初始化采用标准化的 xavier 初始化，激活函数使用 sigmoid
    2. 第 2 个隐藏层：权重初始化采用标准化的 He 初始化，激活函数采用 relu
    3. out 输出层线性层，假若多分类，采用 softmax 做数据归一化

神经网络搭建流程：
    1. 定义一个类，继承：nn.Module
    2. 在 `__init__()` 方法中，搭建神经网络。
    3. 在 `forward()` 方法中，完成：前向传播。

深度学习的四个步骤：
    1. 准备数据。
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试
'''

import torch
import torch.nn as nn
from torchsummary import summary # 查看模型参数有多少
class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        # 输出层
        self.output = nn.Linear(2, 2)

        # 初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # x = self.linear1(x)     # 加权求和
        # x = torch.sigmoid(x)    # 激活函数
        # 合并写法
        x = torch.sigmoid(self.linear1(x))

        x = torch.relu(self.linear2(x))

        x = torch.softmax(self.output(x), dim = -1)

        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_model = ModelDemo().to(device)   # 模型移到设备

    data = torch.randn(size=(5, 3)).to(device)  # 数据也移到同一设备

    output = my_model(data)

    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')


    print(f'output: {output}')
    print(f'output.shape: {output.shape}')
    print(f'output.requires_grad: {output.requires_grad}')

    # 参1：模型对象， 参2：输入数据维度
    summary(my_model, input_size = (3,))

    for name, param in my_model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param} \n')
if __name__ == '__main__':
    train()




















