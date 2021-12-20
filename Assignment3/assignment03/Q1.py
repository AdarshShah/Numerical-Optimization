import numpy as np
from matplotlib import pyplot as plt
# Conjugate Gradient Descent
# Obj  : min 1/2.xAx + cTx
# grad : bx + c

class Q1():
    
    def __init__(self,n) -> None:
        self.n = n
        self.A = np.fromfunction(lambda x,y : 1/(x+y+1),(n,n))
        self.b = -1*np.ones(n)
    
    #Objective Function to minimize
    def obj(self,x:np.ndarray)->float:
        return 0.5*(x.dot(self.A.dot(x))) + self.b.dot(x)

    #Gradient
    def grad(self,x:np.ndarray)->np.ndarray:
        return x.dot(self.A)+self.b

    def ConjugateDescent(self,x:np.ndarray):
        iter=0
        #g0
        g = self.grad(x)
        #u0
        d = -1*g
        error = list()

        #Fletcher-Reeves Implementation
        while np.linalg.norm(g)>=10**-6:
            error.append(np.linalg.norm(g))
            #Alpha
            alpha = -g.dot(d)/d.dot(self.A.dot(d))
            #xk+1
            x = x + alpha*d
            #gk+1
            gk_1 = self.grad(x)
            #Beta
            beta = gk_1.dot(gk_1)/g.dot(g)
            #dk+1
            d = -1*gk_1 + beta*d
            iter+=1
            g = gk_1
        
        error.append(np.linalg.norm(self.grad(x)))
        return x,iter,error

if __name__=='__main__':
    
    for i in [5,8,12,20]:
        problem = Q1(i)
        xopt,iter,error = problem.ConjugateDescent(np.zeros(i))
        print(f'n:{i}\titer:{iter}')

    error = np.log10(error)
    plt.plot(error)
    plt.xlabel('iterations')
    plt.ylabel('log(error)')
    plt.savefig('Q1.jpeg')
