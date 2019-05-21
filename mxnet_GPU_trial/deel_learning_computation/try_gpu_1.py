import mxnet as mx
from mxnet import nd

try:
    ctx = mx.gpu()
    _ =nd.zeros((1, ),ctx=ctx)
except mx.base.MXNetError:
    ctx = mx.cpu()

print(ctx)
