# 3.3
import sys
sys.path.append('D:\python\Lib\site-packages')
from mxnet import nd, autograd
import d2lzh
import random
# 生成原始数据集
num_examples = 1000
num_inputs = 2
true_W = [2, -3.6]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_W[0]*features[ : ,0] +true_W[1]*features[ : ,1]+true_b
labels += nd.random.normal(scale=0.01, shape = labels.shape)
# 读取数据
from mxnet.gluon import data as gdata
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
# 定义模型
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
# 初始化模型参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
# 定义损失函数
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()   # 平方损失又称L2范数损失
# 定义优化算法
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs +1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss %f' % (epoch, l.mean().asnumpy()))
print(true_W)
print(net[0].weight.data())
print(true_b)
print(net[0].bias.data())
