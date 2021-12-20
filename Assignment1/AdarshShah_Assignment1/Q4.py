from math import cos, pi, sin, sqrt
import subprocess
import numpy as np
from numpy import linalg as la

def f(x):
    someVar = subprocess.run(["./getFuncValue_MAC.exe", "19473,["+str(x[0])+","+str(x[1])+"]"],capture_output=True)
    data = str(someVar.stdout)
    return float(data[2:-3])

'''
The main approach is to move around in unit circle.
'''

if __name__=='__main__':
    for theta in np.linspace(-pi,pi,1000):
        x = (cos(theta),sin(theta))
        alpha = (f(x) - x[0] - x[1])
        if alpha < 0:
            print(f'alpha ~{alpha:.6f}. The global minimum does not exist as alpha is not positive at x = ',x)
            break
