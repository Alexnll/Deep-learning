# 3.10
import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'), nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_size = 256