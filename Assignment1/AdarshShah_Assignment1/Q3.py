from math import sqrt
import subprocess
import numpy as np
from numpy import linalg as la

if __name__=='__main__':
    someVar = subprocess.run(["./getDataPoints.exe", "19473"],capture_output=True)
    Y = list()
    X = list()
    data = str(someVar.stdout)
    data = data[2:]
    print(data)
    for s in data.split('\\n'):
        try:
            y,x = s.split(',[')
        except ValueError:
            break
        Y.append(float(y))
        x = x[:-2]
        _X = list()
        for _x in x.split(','):
            _X.append(float(_x))
        _X.append(1)
        X.append(_X)
    Y = np.array(Y)
    Y = np.reshape(Y,newshape=(100,1))
    X = np.array(X)
    xtx = (X.transpose()).dot(X)
    W = (la.inv(xtx)).dot(X.transpose().dot(Y))
    errors = sqrt(np.sum(np.square(np.multiply(Y,-1),X.dot(W))))/100
    print('Least Mean Square Error : ',errors)
    W=np.reshape(W,newshape=(6,))
    for i,w in enumerate(W):
        print(f'{i}: {w:.3f}')