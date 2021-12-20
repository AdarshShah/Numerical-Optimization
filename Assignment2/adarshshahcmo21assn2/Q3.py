import subprocess
import numpy as np
import re
from math import log2

from numpy import linalg
from numpy.linalg.linalg import norm

# Extract data from function(oracle)
process = subprocess.Popen(["./Q3_MAC_oracle.exe"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def read(x:np.ndarray):
    process.stdin.write(f'19473,[{x[0]},{x[1]},{x[2]},{x[3]}]\n'.encode())
    process.stdin.flush()
    data = ""
    for i in range(6):
        data += str(process.stdout.readline())
    numbers = re.findall(pattern="-?\d+.\d+",string=str(data))
    numbers=np.double(numbers)
    fx = numbers[0]
    d2fx = np.zeros(shape=(4,4))
    d2fx[0] = numbers[5:9]
    d2fx[1] = numbers[9:13]
    d2fx[2] = numbers[13:17]
    d2fx[3] = numbers[17:]
    return fx,numbers[1:5],d2fx

#Function f(x) : scalar
def f(x0:np.ndarray)->float:
    x,y,z = read(x0)
    return x

#Gradient g(x) : 1x4
def g(x0:np.ndarray)->np.ndarray:
    x,y,z = read(x0)
    return y

#Hessian h(x) : 4x4
def h(x0:np.ndarray):
    x,y,z = read(x0)
    return z

#Inexact Line Search based on Backtracking
def search(x,iter,a,b,e):
    u=-g(x)
    t=1
    for _ in range(iter):
        if f(x+t*u) <= f(x)+a*t*(g(x).dot(u)):
            return x+t*u,t
        elif t*linalg.norm(u)<e:
            return x+t*u,t
        else:
            t=b*t

#Gradient Descent Algorithm
interf = np.zeros(50)
def gradient_descent(x0,e1,e2,a,b,iter1,iter2):
    x = x0
    i=0
    for _ in range(iter1):
        if abs(norm(g(x))) < e1:
            return x
        x,t = search(x,iter2,a,b,e2)
        interf[i] = f(x)
        i+=1
    return x

#Newton's Method
def newton(x0,e,iter):
    x=x0
    F,G,H = read(x)
    k=0
    for _ in range(iter):
        if abs(norm(G))<e:
            return x,_
        x=x-G.dot(np.linalg.inv(H))
        F,G,H = read(x)
        interf[k] = F
        k+=1
    return x,k

if __name__=='__main__':
    #A
    x,t=search([-1,-1,-1,-1],a=0.5,b=0.5,e=10**-7,iter=30)
    print(f'a : a={abs(log2(t))}, f(x)={f(x):.3f}')
    #B
    x = gradient_descent(x0=[5,-3,-5,3],a=0.5,b=0.5,e1=10**-10,e2=10**-7,iter1=50,iter2=30)
    print(f'b : 1:{interf[0]:.3f},\t5:{interf[4]:.3f},\t10:{interf[9]:.3f},\t20:{interf[19]:.3f},\t50:{interf[49]:.3f}')
    #C
    x,k = newton(x0=[5,-3,-5,3],e=10**-10,iter=50)
    print(f'c : Newton iter : {k},x = {x}')
    process.terminate()


