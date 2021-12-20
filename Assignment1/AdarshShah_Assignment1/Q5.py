import subprocess
import numpy as np
from numpy import linalg as la

# returns fx and gradfx
def get(x):
    someVar = subprocess.run(["./getGradients.exe", "19473,["+str(x[0])+","+str(x[1])+","+str(x[1])+"]"],capture_output=True)
    data = str(someVar.stdout)[2:-3]
    DX = list()
    fx,dX = data.split(',[')
    dX = dX[:-1].split(',')
    {DX.append(float(dX[i])) for i in range(0,3)}
    return float(fx),np.array(DX)


if __name__=='__main__':
    x = np.array((10,10,10))

    iter = 0
    eta = 0.001
    k = 0
    curr,dx = get(x)
    prev = curr+2*eta
    while abs(prev-curr)>eta:
        prev = curr
        x = np.subtract(x,np.divide(dx,k+1))
        curr,dx = get(x)
        k+=1
    
    print('e=0.01 k=',k)


    x = np.array((10,10,10))

    iter = 0
    eta = 0.0001
    k = 0
    curr,dx = get(x)
    prev = curr+2*eta
    while abs(prev-curr)>eta:
        prev = curr
        x = np.subtract(x,np.divide(dx,k+1))
        curr,dx = get(x)
        k+=1
    
    print('e=0.001 k=',k)

    x = np.array((10,10,10))

    iter = 0
    eta = 0.00001
    k = 0
    curr,dx = get(x)
    prev = curr+2*eta
    while abs(prev-curr)>eta:
        prev = curr
        x = np.subtract(x,np.divide(dx,k+1))
        curr,dx = get(x)
        k+=1
    
    print('e=0.0001 k=',k)
