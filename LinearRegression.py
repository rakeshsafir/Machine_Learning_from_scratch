
# coding: utf-8

# In[70]:


import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
t0_array = []
t1_array = []
e_array= []


def gradient_descent(alpha,x,y,ep=0.001, max_iter=1000):
    converged  = False
    iter = 0
    m = len(x) #Number of samples
    t0 = np.random.random(x.shape[1]) #Initial value of theta0
    t1 = np.random.random(x.shape[1]) #Initial value of theta1
    print 'initial theta0 = %s theta1 = %s' %(t0, t1)
    
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)]) #Initial Error
    print 'initial error = %s' %(J)

    while not converged:
        
        grad0 = 1.0/m*(sum([(t0+t1*x[i]-y[i]) for i in range(m)]))
        grad1 = 1.0/m*(sum([(t0+t1*x[i]-y[i])*x[i] for i in range(m)]))
        
        t0 = t0 - alpha * grad0
        t1 = t1 - alpha * grad1
    	print 'For iter = %s theta0 = %s theta1 = %s' %(iter, t0, t1)
        
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )
    	print 'For iter = %s new error = %s' %(iter, e)
        
        # Convergence takes place either 
        # 1)after completing all the iterations or 
        # 2)if error difference current and prev is less than some value here(0.0001)
        
        if abs(J-e)<0.0001:
            print "Converged successfully"
            converged = True
        
        
        if iter==max_iter:
            converged = True
        
        J=e
        iter+=1
	e_array.append(e)
	t0_array.append(t0)
	t1_array.append(t1)

    print "errorfn-theta0 plot"
    plt.scatter(t0_array, e_array)
    plt.show()
    
    print "errorfn-theta1 plot"
    plt.scatter(t1_array, e_array)
    plt.show()
    
    return t0,t1
        
    
#Dummy dataset
x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35) 
print 'x.shape = %s y.shape = %s' %(x.shape, y.shape)

alpha = 0.01
ep = 0.01

print "Initial scatter plot"
plt.scatter(x,y)
plt.show()

theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1000)

print 'theta0 = %s theta1 = %s' %(theta0, theta1)

print "Line plot after finding intercept and slope"

plt.scatter(x,y)
plt.plot(x, theta0+x*theta1, 'r')
plt.show()

