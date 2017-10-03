
# coding: utf-8

# In[45]:

import math
import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_blobs 
import pylab
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def Hypothesis(theta, x, m):
	z = 0
	z = (x[row]*theta for row in range((m)))
	return sigmoid(z)

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in xrange(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i], m)
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in xrange(m):
		xi = X[i]
		hi = Hypothesis(theta,xi, m)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	print 'cost is ', J 
	return J

def gradient_descent(alpha,x,y, max_iter=2000):
    converged  = False
    iter = 0
    m = len(x) #Number of samples
    print m
    t0 = 1.0 #Initial value of theta0
    t1 = 1.0 #Initial value of theta1
    t2 = 1.0 #Initial value of theta1
    print 'initial theta0 = %s theta1 = %s theta2 = %s' %(t0, t1, t2)
    J = 0
#    for row in range(m):
#    	hi = sigmoid(t0*x[row][0]+t1*x[row][1]+t2*x[row][2])
#    	if y[row] == 1:
#    		error = -math.log(hi)
#    	elif y[row] == 0:
#    		error = -math.log(1-hi)
#    	J += error
    J = Cost_Function(x,y,t0,m) 
    print 'initial error = %s' %(J)
#   e = 0	
        
    while not converged:
                
#        grad0 = 1.0/m*( sum([(sigmoid(t0*x[row][0]+t1*x[row][1]+t2*x[row][2])-y[row]) for row in range((m))]))
#        grad1 = 1.0/m*(sum([(sigmoid(t0*x[row][0]+t1*x[row][1]+t2*x[row][2])-y[row])*x[row][1] for row in range((m))]))
#        grad2 = 1.0/m*( sum([(sigmoid(t0*x[row][0]+t1*x[row][1]+t2*x[row][2])-y[row])*x[row][2] for row in range((m))]))
                
        t0 = t0 - (Cost_Function_Derivative(x,y,t0,row,m,alpha) for row in range((m)))
        t1 = t1 - (Cost_Function_Derivative(x,y,t1,row,m,alpha) for row in range((m)))
        t2 = t2 - (Cost_Function_Derivative(x,y,t2,row,m,alpha) for row in range((m)))
    	print 'For iter = %s theta0 = %s theta1 = %s theta2 =  %s' %(iter, t0, t1, t2)
#	for row in range(m):
#		hi = sigmoid(t0*x[row][0]+t1*x[row][1]+t2*x[row][2])
#		if y[row] == 1:
#			error = -math.log(hi)
#		elif y[row] == 0:
#			error = -math.log(1-hi)
#		e += error

        if abs(J-e)>0:
		print 'Converging error = %s' %(e)
    	else:
		print 'Not Converging error = %s' %(e)

        if abs(J-e)<0.0001:
		print "Converged successfully"
		converged = True
        
        if iter==max_iter:
            converged = True
        
        iter+=1
        
    return t0,t1,t2

alpha = 0.001 #learning rate

print "Initial scatter plot"
x,y = make_blobs(n_samples=250, n_features=2, centers=2,cluster_std=1.05, random_state=20)
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y)
plt.show()

#Add new bias column
X = np.c_[np.ones((x.shape[0])), x]
theta0,theta1,theta2 = gradient_descent(alpha, X, y, max_iter=1000)
print 'theta0 = %s theta1 = %s theta2 = %s' %(theta0, theta1, theta2)
Y = (-theta0 -(theta1 * X)) / theta2
print "plot the original data along with our line of best fit"
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")
plt.show()



