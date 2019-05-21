from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(units=10, activation='relu'),
        nn.Dense(units=5)
        )
net.initialize(init=init.Constant(10), force_reinit=True)

X = nd.random.uniform(shape=(2, 20))
net(X)
# Y = net(X)

print(net[0].weight.data()[0])
