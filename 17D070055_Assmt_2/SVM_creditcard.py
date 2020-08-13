import csv
import numpy
import scipy
import math
import pandas as pd

from cvxopt import matrix, solvers

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

x = numpy.delete(m, numpy.s_[29:30], axis=1)


x = numpy.matrix(x)
y = numpy.matrix(y)
#generating G
g = numpy.zeros([200,200])

x = x.astype(numpy.float)
y = y.astype(numpy.float)

#print(x[1])
#print(x[2])
#print(x[1]*((x[2]).T))
y = y.T

for m in range (200):
	for n in range(200):
		g[m,n] = y[m]*y[n]*(x[m]*((x[n]).T))

print(g.shape)
print(y.shape)

P = matrix(g)

G = matrix(numpy.identity(200))
G = -1*G
h =matrix(numpy.zeros([200,1]))

A = matrix(y.T)

b = matrix(numpy.zeros([1,1]))


q = matrix(numpy.ones([200,1]))

sol = solvers.qp(P,q,G,h,A,b)


print(sol['x'])





alpha = numpy.zeros([200,1])
alpha = alpha.astype(numpy.float)
alpha = numpy.matrix(alpha)
#print(type(alpha))

alpha = [[ 3.54e-12],
[ 4.11e-12],
[ 4.23e-12],
[ 6.12e-12],
[ 2.24e-12],
[ 6.31e-12],
[ 4.55e-12],
[ 4.34e-12],
[ 4.18e-12],
[ 4.19e-12],
[ 5.23e-12],
[ 4.04e-12],
[ 4.88e-12],
[ 5.34e-12],
[ 4.21e-12],
[ 4.72e-12],
[ 5.09e-12],
[ 5.17e-12],
[ 4.96e-12],
[ 5.37e-12],
[ 5.71e-12],
[ 5.91e-12],
[ 5.99e-12],
[ 5.94e-12],
[ 1.48e-11],
[ 1.40e-11],
[ 4.15e-12],
[ 7.88e-12],
[ 1.48e-11],
[ 4.30e-12],
[ 7.24e-12],
[ 6.73e-12],
[ 4.71e-12],
[ 2.15e-11],
[ 8.41e-12],
[ 1.04e-11],
[ 1.29e-11],
[ 1.18e-11],
[ 4.08e-12],
[ 2.28e-12],
[ 2.31e-12],
[ 2.33e-12],
[ 1.18e-11],
[ 3.17e-11],
[ 2.84e-11],
[ 1.08e-11],
[ 1.85e-11],
[ 1.21e-11],
[ 1.53e-11],
[ 1.43e-11],
[ 1.38e-11],
[ 1.36e-11],
[ 1.34e-11],
[ 1.30e-11],
[ 1.21e-11],
[ 1.06e-11],
[ 2.26e-12],
[ 1.73e-11],
[ 1.10e-11],
[ 1.11e-11],
[ 2.17e-12],
[ 1.12e-11],
[ 1.14e-11],
[ 1.16e-11],
[ 1.18e-11],
[ 1.20e-11],
[ 1.23e-11],
[ 1.25e-11],
[ 1.27e-11],
[ 1.30e-11],
[ 1.32e-11],
[ 1.35e-11],
[ 1.38e-11],
[ 3.07e-12],
[ 1.32e-11],
[ 7.94e-12],
[ 1.34e-11],
[ 1.36e-11],
[ 1.38e-11],
[ 1.39e-11],
[ 1.41e-11],
[ 2.13e-12],
[ 2.32e-12],
[ 3.77e-12],
[ 2.16e-12],
[ 2.64e-12],
[ 3.24e-12],
[ 6.93e-12],
[ 2.49e-12],
[ 2.41e-12],
[ 3.35e-12],
[ 6.23e-12],
[ 6.33e-12],
[ 2.22e-12],
[ 5.82e-12],
[ 6.01e-12],
[ 5.98e-12],
[ 6.61e-12],
[ 6.14e-12],
[ 4.24e-12],
[ 5.80e-12],
[ 8.48e-12],
[ 5.43e-12],
[ 9.08e-12],
[ 8.69e-12],
[ 8.20e-12],
[ 9.74e-12],
[ 5.83e-12],
[ 7.88e-12],
[ 8.15e-12],
[ 9.38e-12],
[ 7.60e-12],
[ 8.29e-12],
[ 9.15e-12],
[ 7.35e-12],
[ 8.31e-12],
[ 8.46e-12],
[ 7.93e-12],
[ 8.26e-12],
[ 1.23e-11],
[ 7.52e-12],
[ 1.05e-11],
[ 8.68e-12],
[ 8.18e-12],
[ 8.92e-12],
[ 8.65e-12],
[ 7.53e-12],
[ 7.86e-12],
[ 8.24e-12],
[ 8.78e-12],
[ 8.96e-12],
[ 9.54e-12],
[ 9.76e-12],
[ 9.76e-12],
[ 9.77e-12],
[ 9.77e-12],
[ 7.70e-12],
[ 1.06e-11],
[ 1.59e-11],
[ 9.92e-12],
[ 9.57e-12],
[ 1.19e-11],
[ 8.52e-12],
[ 7.69e-12],
[ 8.24e-12],
[ 8.68e-12],
[ 7.22e-12],
[ 9.01e-12],
[ 8.35e-12],
[ 8.80e-12],
[ 9.16e-12],
[ 2.76e-12],
[ 7.96e-12],
[ 7.90e-12],
[ 7.44e-12],
[ 8.86e-12],
[ 7.77e-12],
[ 9.89e-12],
[ 9.70e-12],
[ 8.34e-12],
[ 7.29e-12],
[ 7.46e-12],
[ 6.24e-12],
[ 8.05e-12],
[ 6.48e-12],
[ 1.03e-11],
[ 6.97e-12],
[ 5.74e-12],
[ 3.08e-11],
[ 7.82e-12],
[ 9.11e-12],
[ 9.49e-12],
[ 8.34e-12],
[ 8.76e-12],
[ 1.19e-11],
[ 1.03e-11],
[ 8.85e-12],
[ 7.25e-12],
[ 8.18e-12],
[ 8.44e-12],
[ 7.53e-12],
[ 9.93e-12],
[ 7.51e-12],
[ 6.42e-12],
[ 7.88e-12],
[ 7.46e-12],
[ 7.18e-12],
[ 9.39e-12],
[ 7.22e-12],
[ 2.97e-12],
[ 7.56e-12],
[ 6.10e-12],
[ 9.02e-12],
[ 7.50e-12],
[ 9.20e-12],
[ 7.72e-12],
[ 8.77e-12],
[ 8.35e-12],
[ 6.02e-12],
[ 7.32e-12]]

#print(x.shape)

w =  numpy.zeros([1,29])

u =  numpy.zeros([1,29])

for i in range(200):
	u = alpha[n]*y[n]*x[n]
	w = w+u


#print(w.shape)

dum1 = x*w.T
dum2 = dum1[100:]
dum3 = dum1[:100]

k1= min(dum3)+max(dum2)
b1= numpy.zeros([200,1])
b1 =-0.5*(k1)+b1

y1 = x*w.T+(b1)



w = numpy.matrix(w)
w = w.astype(numpy.float)


fin = numpy.zeros([200,1])
#print((y1))
for d in range(200):
	if (y1[d]>1.0):
		fin[d]=-1.0
	elif (y1[d]<1.0) :
		fin[d]=1.0

#print(fin)



#loading test data



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
y2 = m1[:,-1]
x1 = numpy.delete(m1, numpy.s_[29:30], axis=1)





x1 = numpy.matrix(x1)
y2 = numpy.matrix(y2)
y2 = y2.T

x1 = x1.astype(numpy.float)
y2 = y2.astype(numpy.float)


b2= numpy.zeros([40,1])
b2 =-0.5*(k1)+b2


y3 = x1*w.T +b2
y3 = y3.astype(numpy.float)
#print(y3)
avg = numpy.ones([1,40])*y3/40.0
#print(avg)
fin1 = numpy.zeros([40,1])
for d2 in range(40):
	if ((y3[d2])<9e-05):
		fin1[d2]=-1
	else:
		fin1[d2]=1
err = fin1-y2
#print(err)
correct = 0

for o in range(40):
	if err[o]==0:
		correct= correct+1
	else:
		correct = correct

acuracy = correct/40.0

print('Accuracy in Detection of fraud is', 100.0*acuracy,'%')
