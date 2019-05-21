from mxnet import nd
from mxnet.gluon import nn

# x = nd.ones(3)
# nd.save('x', x)

# print(nd.load('x'))
'''
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize()
X = nd.random.uniform(shape=(5,10))
y = net(X)
filename = 'net.params'
net.save_parameters(filename)
nd.save('y',y)
nd.save('x',X)
'''
filename = 'net.params'
net2 = nn.Sequential()
net2.load_parameters(filename)
X = nd.load('x')
Y2 = net2(X)
y = nd.load('y')
print(y == Y2)
