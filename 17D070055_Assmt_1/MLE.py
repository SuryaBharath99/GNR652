
import csv
import numpy
import scipy
import math
import pandas as pd
import matplotlib.pyplot as plt




m = numpy.array([[]])
with open('data.csv','r') as f:
	 data = csv.reader(f,delimiter=',')
	 first =0
	 for row in data:
	 	if first==0:
	 		first=1
	 		continue
	 	if first==1:
	 		m=numpy.array([row])
	 		first = 2
	 		continue
	 	else:
	 	 m= numpy.append(m,[row],axis=0)


data = m

y = m[:,-1]
x = numpy.delete(m, numpy.s_[17:18], axis=1)
z = numpy.ones((108,1))
x = numpy.append(z,x,axis=1)

w = numpy.ones((18,1))

iter =2000
alpha = 0.001
x = numpy.matrix(x)
y = numpy.matrix(y)
z = numpy.matrix(z)
w = numpy.matrix(w)

def matrixmult (A, B):
    C = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k]*B[k][j]
    return C


#def cost (X,Y,W):
#	return (numpy.array((matrixmult(W, W.T)))*0.01)+numpy.array(matrixmult((matrixmult(X, W) - Y), (matrixmult(X, W) - Y).T))*0.00463

x = x.astype(numpy.float)
y = y.astype(numpy.float)
#u = (w.T * x.T)
#u = u.T
y = y.T
#u1 = u-y
#cost = ((w.T * w)/100) +((u1.T*u1)/216)





for i in range(iter) :

    u = (w.T * x.T).T

    u1 =(((w.T * x.T).T) - y)

    w = w-((0.004/108)*(0.01*2*w  +  ((u1.T * x).T)))

    k = u1






m1 = numpy.array([[]])
with open('test.csv','r') as f1:
	 data = csv.reader(f1,delimiter=',')
	 first1 =0
	 for row in data:
	 	if first1==0:
	 		first1=1
	 		continue
	 	if first1==1:
	 		m1=numpy.array([row])
	 		first1 = 2
	 		continue
	 	else:
	 	 m1= numpy.append(m1,[row],axis=0)

test = m1
y1 = m1[:,-1]
x1 = numpy.delete(m1, numpy.s_[17:18], axis=1)

z1 = numpy.ones((27,1))
x1 = numpy.append(z1,x1,axis=1)


x1 = numpy.matrix(x1)
y1 = numpy.matrix(y1)
z1 = numpy.matrix(z1)

x1 = x1.astype(numpy.float)
y1 = y1.astype(numpy.float)


u3 = ((w.T)*x1.T).T
y1 = y1.T

l = u3-y1
Test_cost_value =(l.T * l)/(2*27)
print('Mean Squared Error of TEST data in MLE method is',Test_cost_value)
