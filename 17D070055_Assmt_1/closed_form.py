import csv
import numpy
import scipy
import math
import pandas as pd


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

x = numpy.matrix(x)
y = numpy.matrix(y)
z = numpy.matrix(z)
w = numpy.matrix(w)

x = x.astype(numpy.float)
y = y.astype(numpy.float)


i = x.T * x


I = numpy.identity(18)


M = i+(0.01*I)
Minv = numpy.linalg.inv(M)


k = x.T * y.T

w = Minv * k


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

u2 = ((w.T)*x1.T).T
y1  = y1.T

l = u2-y1

Test_cost_value =(l.T * l)/(2*27)
print('Mean Squared Error of TEST data in Closed form method  is',Test_cost_value)
