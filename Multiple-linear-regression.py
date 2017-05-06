from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D


def gradientDescent(x, y, theta, alpha, m, numIterations):
    J_history = np.zeros(shape=(numIterations, 1))
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta) #calculates the hypothesis by multiplying the parameter theta with input 
        loss = hypothesis - y #calculats the difference between the hypothesis and the actual value
   
        cost = np.sum(loss ** 2) / (2 * m) #cost function formula
        print("Iteration %d | Cost: %f" % (i, cost)) #gradient descent per iteration is displayed
        gradient = np.dot(xTrans, loss) / m #evaluates diffirential of the cost function
        # update
        theta = theta - alpha * gradient #estimates new theta with 
        J_history[i][0] = cost 
    return theta, J_history

a = [
[1,0,0],[1,0,0],
]        #add the number of items in your sample set and number of zeros as the number of variables, this has two zeros for two variables

X1 = [[],[]] #input values for first variable

X2 = [[],[]] #input values for second variable

"""
As many lists are created as the number of variables , here two variables has been set

"""



a[0][1] = X1[0][0]
a[1][1] = X1[1][0] 

a[0][2] = X2[0][0]
a[1][2] = X2[1][0]

""" Here the input values are being stored in the matrix A in the secnd and third columns"""

for x in a:
	print x
	print ""



b = [[],[]] #store results here 

for y in b:
	print y
	print ""

print "X Matrix"
x = np.asarray(a) 
print x
print "" #prints X matrix
print "Y Matrix"
y = np.asarray(b)
print y
print "" #prints Y matrix

m, n = np.shape(x)
numIterations= 1000000 #number of times you want gradient descent to run
alpha = 0.01           #your learning rate
theta = np.ones(n)

theta, J_history = gradientDescent(x, y, theta, alpha,m,numIterations)
print "theta"
print(theta)
print ""
print "hx"
hx = x.dot(theta) #Hypothesis is calculated with the theta generated from the previous gradient descent
print hx
print ""
print "Difference"
diff = hx - y
print diff
print ""
print "Difference Square"
diff_square = diff*diff
print diff_square 
print ""
print "Sum"
sum = np.sum(diff_square) 
print sum
print ""
print "Cost function"
#cost = 1/(2*m)*sum(diff_square)
temp = 1/(2*m)
cost = temp*sum
print cost #cost function is calculated for that particular theta

print theta, J_history
plot(np.arange(numIterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()            

"""
The above code can be used to verify that the cost function is decreasing as gradient descent runs
"""

print theta
meany = np.mean(y)
print meany

sumsqmeany = np.sum((y - meany)**2)
print sumsqmeany

sumsqmeanysum = np.sum((y - hx)**2)/sumsqmeany

R = 1 - sumsqmeanysum
print "The R value is:"
print R 

"""
The above code is used to calculate the coefficient of determination - which is used to explain how well the generated model explains the data

"""
answer = np.array([1.0,  X1(TEST), X2(TEST) ]).dot(theta)
print 'answer is : %f' % (answer)

"""
Here in place of X1(TEST) and X2(TEST) enter your new input belonging to a training or testing data set to predict the output
"""



