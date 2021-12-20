import subprocess
import numpy as np
import re

from numpy import linalg
from numpy.linalg import norm

# Extract data from function(oracle)
process = subprocess.Popen(["./Q6_MAC_oracle.exe"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def read(x:np.ndarray):
    process.stdin.write(f'19473,[{x[0]},{x[1]},{x[2]},{x[3]},{x[4]},{x[5]},{x[6]},{x[7]},{x[8]},{x[9]}]\n'.encode())
    process.stdin.flush()
    data = ""
    for i in range(3):
        data += str(process.stdout.readline())
    numbers = re.findall(pattern="-?\d+.\d+",string=str(data))
    numbers=np.double(numbers)
    return numbers[0],numbers[1:11],numbers[11:]

#f(x) : 1x1
def f(x:np.ndarray)->np.ndarray:
    a,b,c = read(x)
    return a

#gradient f(x) : 10x1
def grad(x:np.ndarray)->np.ndarray:
    a,b,c = read(x)
    return b

#Qx : 10x1
def Qx(x:np.ndarray)->np.ndarray:
    a,b,c = read(x)
    return c


#gradient descent for 6.f iii
def gradient_descent(x0,e):
    x = x0
    i = 1
    u = -grad(x)
    g = grad(x)
    while True:
        #StepSize
        a = -u.dot(g)/(Qx(u).dot(u))
        x = x + a*u
        #Stopping Condition xk+1 - xk = a*u <= e
        if abs(norm(a*u)) <= e:
            return x,i
        u = - grad(x)
        g = grad(x)
        i+=1

#gradient descent for 6.f iv
def gradient_descent2(x0,e,b):
    SST = np.identity(10)
    SST[0][0] = b
    x = x0
    i = 1
    u = -grad(x).dot(SST)
    g = grad(x)
    while True:
        #StepSize
        a = -u.dot(g)/(Qx(u).dot(u))
        x = x + a*u
        #Stopping Condition xk+1 - xk = a*u <= e
        if abs(norm(a*u)) <= e:
            return x,i
        u = - grad(x).dot(SST)
        g = grad(x)
        i+=1

if __name__=="__main__":
    beta = [1/200,1/700,1/2000]
    x0 = np.ones(10)*50
    x0[0]=1
    #Problem 6.f ii Number of iterations by gradient descent
    x,iter = gradient_descent(x0,0.01)
    print(f'iter : {iter}')

    #Problem 6.f ii Number of iterations by gradient descent
    for b in beta:
        x,iter = gradient_descent2(x0,0.01,b)
        print(f'iter : {iter}, x : {x}')
    
    process.terminate()
    