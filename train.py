i = 0
j = 0
first = 0
second = 0
X = [1,3,5,7,9]
Y = [2,4,6,8,10]
n = len(X)
for count in range(0, 10):
    if i+j == n+1:
        break
    if X[i]<Y[j]:
        first=second
        second=X[i]
        i = i + 1
    else:
        first=second
        second = Y[j]
        j = j + 1
print((first+second)/2)