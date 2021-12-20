import numpy as np
from matplotlib import pyplot as plt

SRno = 19473

def gradientDescent(x0,A,b,z,step,iter):
    #Obtaining L
    _,s,_ = np.linalg.svd(A)
    L = s[0]**2

    x=x0
    result = np.zeros((iter,x0.shape[0]))
    for i in range(iter):
        x = x - step*(A@A.transpose()@x + b - A@z)/L
        x = np.maximum(0,x)
        result[i,:] = x

    return result

def gradientDescent2(x0,A,b,z,step,iter):
    #Obtaining Identity Matrix

    B = A.transpose()

    x=x0
    for i in range(iter):
        x = x - step*(x-z) + B@np.linalg.inv(A@B)@b

    return x

if __name__=="__main__":
    #Q5.c
    iter=100
    step = 1 # 1/L
    A = np.array([[1,1,1],[-1,0,0],[0,-1,1]])
    b = np.array([3,0,0])
    z = np.array([3,-1,2])
    
    lamda0 = np.array([1,1,1])
    result = gradientDescent(lamda0,A,b,z,step,iter)

    x = np.zeros((iter,z.shape[0]))
    for i in range(iter):
        x[i,:] = z - A.transpose()@result[i,:]
    

    #Q5.d i
    iter = 10
    step = 1  # 1/L
    A = np.array([[1,1],[-1,0],[0,-1]])
    b = np.array([1,0,0])
    z = np.array([2,1])
    
    lamda0 = np.array([1,1,1])
    result = gradientDescent(lamda0,A,b,z,step,iter)

    x = np.zeros((iter,z.shape[0]))
    for i in range(iter):
        x[i,:] = z - A.transpose()@result[i,:]
    
    #Plotting
    X = [0,1,0,0]
    Y = [0,0,1,0]
    plt.plot(X,Y,color='blue')
    plt.plot(x[:,0],x[:,1],color='red')
    plt.savefig("Q5_d_i")
    plt.cla()

    #Q5.d ii
    iter = 10
    step = 2  # 1/L
    A = np.array([[1,1],[-1,0],[0,-1]])
    b = np.array([1,0,0])
    z = np.array([2,1])
    
    lamda0 = np.array([1,1,1])
    result = gradientDescent(lamda0,A,b,z,step,iter)

    x = np.zeros((iter,z.shape[0]))
    for i in range(iter):
        x[i,:] = z - A.transpose()@result[i,:]
    
    #Plotting
    X = [0,1,0,0]
    Y = [0,0,1,0]
    plt.plot(X,Y,color='blue')
    plt.plot(x[:,0],x[:,1],color='red')
    plt.savefig("Q5_d_ii")
    plt.cla()

    #Q5.e
    step = 1  # 1/L
    A = np.array([[1,3,0],[0,2,1]])
    b = np.array([1,-1])
    z = np.array([0,0,0])
    
    x0 = np.array([2.5,-0.5,0])
    result = gradientDescent2(x0,A,b,z,step,100)
    print(result)
    