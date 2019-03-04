# 9.1
from mxnet import image
import d2lzh as d2l
import matplotlib.pyplot as plt

d2l.set_figsize()
img = image.imread('D:/picture/55541084_p0.jpg')
d2l.plt.imshow(img.asnumpy())
