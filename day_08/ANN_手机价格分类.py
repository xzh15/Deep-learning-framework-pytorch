"""
ANN案例的实现步骤:
1. 构建数据集.
2. 搭建神经网络.
3. 模型训练.
4. 模型测试.
"""

# 导包
import torch                        # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset  # 数据集对象. 数据 → Tensor → 数据集 → 数据加载器
from torch.utils.data import DataLoader     # 数据加载器.
import torch.nn as nn               # neural network, 封装了神经网络的各种操作
import torch.optim as optim         # 优化器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
import matplotlib.pyplot as plt     # 绘图
import numpy as np                  # 数组(矩阵)操作
import pandas as pd                 # 数据处理
import time                         # 时间模块

from torchsummary import summary

# todo 1. 定义函数, 构建数据集.
def create_dataset():
    data = pd.read_csv('data/train.csv')
    print(f'data:{data.head()}')
    print(f'shape:{data.shape}')

    # 2. 获取x特征列 和 y标签列.
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f'x: {x.head()}, {x.shape}')  # (2000, 20)
    # print(f'y: {y.head()}, {y.shape}')  # (2000, )

    # 3. 把特征列转成浮点型.
    x = x.astype(np.float32)
    # print(f'x: {x.head()}, {x.shape}')  # (2000, 20)

    # 4. 切分训练集和测试集.
    # 参1: 特征, 参2: 标签, 参3: 测试集所占比例, 参4: 随机种子, 参5: 样本的分布(即: 参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 5. 把数据集封装成 张量数据集. 思路: 数据 → 张量Tensor → 数据集TensorDataSet → 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    # 6. 返回结果
    # 20(充当 输入特征数)      4(充当 输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# todo 2. 搭建神经网络.
class PhonePriceModel(nn.Module):
    # 1. 在init魔法方法中, 初始化父类成员, 及搭建神经网络.
    def __init__(self, input_dim, output_dim):  # 输入: 20, 输出: 4
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 搭建神经网络.
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        out = self.output(x)
        return out

# todo 3. 模型训练.
def train(train_dataset, input_dim, output_dim):
    # 1. 创建数据加载器, 流程: 数据 → 张量 → 数据集 → 数据加载器
    # 参1: 数据集对象(1600条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 2. 创建神经网络模型.
    model = PhonePriceModel(input_dim, output_dim)
    # 3. 定义损失函数, 因为是多分类, 这里用的是: 多分类交叉熵损失函数.
    criterion = nn.CrossEntropyLoss()
    # 4. 创建优化器对象.
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 5. 模型训练.
    # 5.1 定义变量, 记录训练的总轮数.
    epochs = 300
    # 5.2 开始(每轮的)训练.
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录每次训练的损失值, 训练批次数.
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量, 表示训练开始的时间.
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练.
        for x, y in train_loader:
            # 5.2.4 切换模型(状态)
            model.train()  # 训练模式.    model.eval()   # 测试模式.
            # 5.2.5 模型预测.
            y_pred = model(x)
            # 5.2.6 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零, 反向传播, 优化参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.8 累加损失值.
            total_loss += loss.item()  # 把本轮的每批次
            batch_num += 1

        # 5.2.9 打印训练结果.
        print(f'epoch: {epoch+1}, loss: {total_loss/batch_num}, time: {time.time()-start}')
    # 保存模型
    print(f'\n\n模型参数: {model.state_dict()}\n\n')

    torch.save(model.state_dict(), './model/phone.pth')
# todo 4. 模型测试.
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络分类对象.
    model = PhonePriceModel(input_dim, output_dim)
    # 2. 加载模型参数.
    model.load_state_dict(torch.load('./model/phone.pth'))
    # 3. 创建测试集的 数据加载器对象.
    # 参1: 数据集对象(400条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 4. 定义变量, 记录预测正确的样本个数.
    correct = 0
    # 5. 从数据加载器中, 获取到每批次的数据.
    for x, y in test_loader:
        # 5.1 切换模型状态 → 测试模式.
        model.eval()
        # 5.2 模型预测.
        y_pred = model(x)
        print(f'y_pred: {y_pred}')  # [[0分类概率, 1分类概率, 2分类概率, 3分类概率],
        # 5.3 根据加权求和, 得到类别, 用argmax()获取最大值对应的下标, 就是类别.
        y_pred = torch.argmax(y_pred, dim=1)  # dim=1 表示逐行处理.
        print(f'y_pred: {y_pred}')  # [第1条数据的预测分类, ...]
        # 5.4 统计预测正确的样本个数.
        # print(y_pred == y)
        # print((y_pred == y).sum())
        correct += (y_pred == y).sum()

    # 6.走到这里, 模型预测结束, 打印准确率即可.
    print(f'准确率(Accuracy): {correct / len(test_dataset):.4f}')


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'训练集 数据集对象: {train_dataset}')
    print(f'测试集 数据集对象: {test_dataset}')
    print(f'输入特征数: {input_dim}')    # 20
    print(f'输出标签数: {output_dim}')  # 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhonePriceModel(input_dim, output_dim).to(device)
    summary(model, input_size=(input_dim,))
    #train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)




