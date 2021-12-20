import subprocess
import numpy as np
from math import log

#The univariate functions to minimize
f = lambda x : (1+x**2)**0.5
g = lambda x : x/(1+x**2)**0.5
h = lambda x : 1/(1+x**2)**0.5*(1-x**2*(1+x**2)**-0.5)

#Newton's Method
def newton(x0,e):
    x=x0
    F,G,H = f(x),g(x),h(x)
    k=0
    while abs(G)>e:
        x=x-G/H
        F,G,H = f(x),g(x),h(x)
        k+=1
    return x,k

#The univariate functions to minimize
f = lambda x : log(1+x**2)/2
g = lambda x : x/(1+x**2)
h = lambda x : (1 - x**2)/(x**2 + 1)**2

if __name__=='__main__':
    X = [0.1,0.6,-0.5,1.2]
    for x in X:
        ans,k = newton(x0=x,e=0.05)
        print(f'Q5 : X:{x:.1f}, X*:{ans:.3f}, iter:{k},f(x) = {f(x)}, g(x) = {g(x)}')