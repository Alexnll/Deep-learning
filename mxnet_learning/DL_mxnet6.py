# 3.9
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

number_inputs, number_outputs, number_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01,shape=(number_inputs, number_hiddens))
b1 = nd.zeros(shape=number_hiddens)
W2 = nd.random.normal(scale=0.01,shape=(number_hiddens, number_outputs))
b2 = nd.zeros(shape=number_outputs)

params=[W1,b1,W2,b2]
for param in params:
    param.attach_grad();

def relu(X):
    return nd.maximum(X,0)
def net(X):
    X = X.reshape((-1, number_inputs))
    H = relu(nd.dot(X, W1)+ b1)
    return nd.dot(H, W2)+b2

loss = gloss.SoftmaxCrossEntropyLoss()

number_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter,test_iter,loss, number_epochs, batch_size, params, lr)
print(W1)
print(b1)
print(W2)
print(b2)