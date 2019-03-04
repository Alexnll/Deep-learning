# 3.2
import sys
sys.path.append('D:\python\Lib\site-packages')
from mxnet import nd, autograd
from IPython import display
from matplotlib import pyplot as plt
import random
# 读取批量数据，大小为batch_size
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)

# 生成原始数据集
num_examples = 1000
num_inputs = 2
true_W = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_W[0]*features[ : ,0] +true_W[1]*features[ : ,1]+true_b
labels += nd.random.normal(scale=0.01, shape = labels.shape)
# 确定批量大小

# 初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs,1))
b = nd.zeros(shape=1)
w.attach_grad()
b.attach_grad()

# 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b
# 定义损失函数
def squared_loss(y_hat, y):
    return(y_hat - y.reshape(y_hat.shape))**2/2
# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad/batch_size

# 模型训练
# 定义超参数
lr = 0.05           # learning rate
num_epochs = 10       # 迭代周期
batch_size = 10       # 批量大小
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_l.mean().asnumpy()))

print(true_W)
print(w)
print(true_b)
print(b)
'''
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
set_figsize()
'''
# 可视化
# plt.scatter(features[: ,1].asnumpy(), labels.asnumpy(),1)
# plt.show()