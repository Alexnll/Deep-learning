import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import time
# print(mx.cpu(0), mx.gpu(),mx.gpu(1))

x = nd.random.uniform(shape=(100, 2000), ctx=mx.gpu())
z = nd.random.uniform(shape=(2000, 300),ctx=mx.gpu())
for i in range(10):
    a = nd.dot(x,z)
    time.sleep(1)
    print(i)
print(a.shape, a.context)

