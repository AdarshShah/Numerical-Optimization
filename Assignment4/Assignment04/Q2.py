import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

angles = np.fromfile('a4q2_headingangles.csv',sep=',')

def LP(xf,yf,vmin,fig):
    #Generating Linear Program Constraints
    A = np.zeros((2,100))
    A[0,:] = np.cos(angles)
    A[1,:] = np.sin(angles)

    b = np.zeros(2)
    b[0] = xf - vmin*np.sum(np.cos(angles))
    b[1] = yf - vmin*np.sum(np.sin(angles))

    #Objective
    c = -1*np.ones(100)

    #Optimizing
    res = linprog(c,A_eq=A,b_eq=b)

    #Velocities
    v = res.x + vmin

    #Positions
    x = np.cos(angles)*v
    y = np.sin(angles)*v

    for i in range(1,100):
        x[i],y[i] = x[i]+x[i-1],y[i]+y[i-1]
    
    #Plot
    plt.scatter(x,y)
    plt.savefig(f'q2_{fig}')
    plt.cla()

if __name__=="__main__":
    LP(xf=3,yf=4,vmin=0.01,fig=1)
    LP(xf=3,yf=4,vmin=0,fig=2)
    #Infeasible
    LP(xf=3,yf=4,vmin=1,fig=3)
    #theta f=pi/2
    angles[-1]=np.pi/2
    LP(xf=3,yf=4,vmin=0.01,fig=4)


