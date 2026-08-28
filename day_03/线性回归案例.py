'''
    numpy对象 --> 张量Tensor --> 数据集对象TensorDataset --> 数据加载器DataLoader

    模型构建流程：
    1. 准备训练集数据
    2. 构建要使用的模型
    3. 设置损失函数和优化器
    4. 模型训练

'''

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset():
    x, y, coef = make_regression(
        n_samples= 100,
        n_features = 1,
        noise = 10,
        coef = True,
        random_state = 3
    )

    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)

    return x, y, coef

def train(x, y, coef):
    # 1.创建数据集对象， 把 tensor --> 数据集对象 --> 数据加载器
    dataset = TensorDataset(x, y)
    # 2.创建数据加载器对象
    # 参1： 数据集对象， 参2：批处理大小， 参3：是否打乱数据
    dataloader = DataLoader(dataset, batch_size= 16, shuffle = True)
    # 3.创建初始的线性回归模型
    # 参1： 输入特征个数， 参2： 输出特征个数
    model = nn.Linear(1, 1)
    # 4.创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    # 5.创建损失函数对象
    criterion = nn.MSELoss()
    # 6.具体训练过程
    # 6.1定义变量，分别表示：训练轮数， 每轮的（平均）损失值，训练总损失值， 训练的样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0, 0
    # 6.2开始训练，按轮训练
    for epoch in range(epochs):
        # 6.3 每轮是分批次训练的， 所以从数据加载器中获取批次数据
        for train_x, train_y in dataloader:
            # 6.4 模型预测
            y_pred = model(train_x)
            # 6.5 计算损失值
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            # 6.6计算总损失 和 样本批次数
            total_loss += loss.item()
            total_sample += 1
            # 6.7梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 6.8 把本轮的平均损失值添加到列表中
        loss_list.append(total_loss/total_sample )
        print(f'第{epoch+1}轮训练， 损失值为：{total_loss / total_sample}')

    # 7.最终训练结果
    print(f'{epochs}轮的平均损失分别是：{loss_list}')
    print(f'模型的权重为：{model.weight}')
    print(f'模型的偏置为：{model.bias}')

    # 8.绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()
    plt.show()

    # 9 绘制预测值和真实值的关系
    # 9.1 绘制样本点分布情况
    plt.scatter(x, y)
    # 9.2 绘制训练模型的预测值
    # x : 100个样本点的特征
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    # 9.3 计算真实值
    y_true = torch.tensor(data = [v * coef + 0.0 for v in x])
    #9.4 绘制预测值 和 真实值的折线图
    plt.plot(x, y_pred, color = 'red', label = '预测值')
    plt.plot(x, y_true, color = 'blue', label = '真实值')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    x, y, coef = create_dataset()
    train(x, y, coef)

