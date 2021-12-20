import numpy as np

class Q2():

    def __init__(self,sr) -> None:
        self.Q = np.zeros((6,6))
        for i in range(6):
            self.Q[i][i]=sr%100+10-2*i
    
    def f(self,x):
        return 0.5*x.dot(self.Q.dot(x))

    def gradientDescent(self,x,iter):
        g = self.Q.dot(x)
        #Steepest Gradient as descent
        d = -1*g
        result = np.zeros(iter)
        for _ in range(iter):
            #Exact Line Search
            a = -1*g.dot(d)/d.dot(self.Q.dot(d))
            x = x + a*d
            result[_] = self.f(x)
            g = self.Q.dot(x)
            d = -1*g
        return result


    #Inexact Line Search based on Backtracking
    def inexactSearch(self,x,u,a,b,e):
        g=self.Q.dot(x)
        t=1
        while True:
            if self.f(x+t*u) <= self.f(x)+a*t*(g.dot(u)):
                return x+t*u
            elif np.linalg.norm(t*u)<e:
                return x+t*u
            else:
                t=b*t

    def DFP(self,x,method,iter):
        B = np.identity(6)
        result = np.zeros(iter)
        for _ in range(iter):
            g = self.Q.dot(x)
            d = -B.dot(g)
            
            if method=='exact':
                xk_1 = x + (-1*g.dot(d))/d.dot(self.Q.dot(d))*d
            elif method=='backtrack':
                xk_1 = self.inexactSearch(x,d,0.5,0.5,10**-7)
            
            gk_1 = self.Q.dot(xk_1)

            delta = xk_1-x
            gamma = gk_1-g

            dell = delta.reshape((6,1))
            gamm = gamma.reshape((6,1))
            
            B = B + dell@np.transpose(dell)/delta.dot(delta) - B@gamm@np.transpose(gamm)@B/gamma.dot(B.dot(gamma))

            x = xk_1
            result[_] = self.f(x)
        return result


if __name__=='__main__':

    problem = Q2(19473)

    print(problem.gradientDescent(10*np.ones(6),6))

    print(problem.DFP(10*np.ones(6),'exact',6))

    print(problem.DFP(10*np.ones(6),'backtrack',6))




