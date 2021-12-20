import subprocess
import numpy as np
from math import log

#Inexact Line Search based on Backtracking
def search(x,iter,a,b,e):
    u=-g(x)
    t=1
    for _ in range(iter):
        if f(x+t*u) <= f(x)+a*t*g(x)*u:
            return x+t*u,t
        elif abs(t*u)<e:
            return x+t*u,t
        else:
            t=b*t

#Gradient Descent Algorithm
interf = np.zeros(50)
def gradient_descent(x0,e1,e2,a,b,iter):
    x = x0
    k=0
    while abs(g(x)) > e1:
        x,_ = search(x,iter,a,b,e2)
        k+=1
    return x,k

#The univariate functions to minimize
f = lambda x : log(1+x**2)
g = lambda x : x/(1+x**2)
h = lambda x : (1 - x**2)/(x**2 + 1)**2

if __name__=='__main__':
    X = [0.1,0.6,-0.5,1.2]
    for x in X:
        ans,k = gradient_descent(x0=x,a=0.5,b=0.1,e1=0.05,e2=0.01,iter=30)
        print(f'Q4 : X:{x:.1f}, X*:{ans:.3f}, iter:{k}')