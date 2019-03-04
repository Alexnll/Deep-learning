import sys
sys.path.append('D:\python\Lib\site-packages')
import numpy as np
import matplotlib.pyplot as plt

# define Sigmoid function
def g_function(z,n=0):
    if n == 1:
        return z*(1-z)
    return 1/(1+np.exp(z))

# input,利用np.array讲list转变为array
X = np.array([1.5,0.7,0.9])
# output
y = np.array([0.5])
#初始化权重矩阵
w1 = np.array([[0.1,0.3,0.5,0.7],
              [0.2,0.4,0.6,0.8],
              [0.3,0.5,0.7,0.9]])
w2 = np.array([0.3,0.4,0.7,0.9])
# 添加偏置
X_b = np.append(np.array([1]),X)
# 开始迭代
for i in np.arange(100):
    print(i)
# 前向传播
    a0 = X_b
    z1 = np.dot(w1,a0)
    a1 = g_function(z1)
    z2 = np.dot(w2, np.append(np.array([1]),a1))
    hx = g_function(z2)
    print("hx="+str(hx))
    plt.scatter(i, hx)
# 后向传播
    l2_error = hx - y
    print("l2_error="+str(l2_error))
    l2_delta = g_function(hx,1)*l2_error*a1
    l1_error = w2*l2_error
    l1_delta = g_function(a1,1)*l1_error*X

# 修改权重
    w2 = w2 + 0.5*l2_delta
    w1 = w1 + 0.5*l1_delta
    print("w1: "+str(w1),'\n',"w2: "+str(w2))
    if i == 99:
        plt.show()
