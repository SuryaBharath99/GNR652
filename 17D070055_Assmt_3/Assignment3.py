from scipy.io import loadmat
import numpy as np
import math
import csv
import numpy
import scipy
import math
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

X_data=loadmat('Indian_pines_corrected.mat')
X_data=X_data['indian_pines_corrected']
x_data=X_data.copy()


Y_data=loadmat('Indian_pines_gt.mat')

Y_data = Y_data['indian_pines_gt']


y_data=Y_data.copy()


x_data = np.resize(x_data,(21025, 200))


y_data = np.resize(y_data,(21025,1))

dum = np.zeros((200,1))

dum1 = np.zeros((1,1))
y_data = y_data
x_data = x_data.T


print('Please wait...Getting the data ready')
#print((dum3.shape))
for i in range (21025):
    if y_data[i-1,0] != 0 :
        l = x_data[:, i-1]
        k = np.resize(l, (200, 1))
        l1 = y_data[i-1, :]
        k1 = np.resize(l1, (1, 1))
        dum  = np.append(dum, k, axis=1)
        dum1 = np.append(dum1, k1, axis=1)
    else:
        dum = dum
        dum1=dum1

x_cleaned = dum[:,1:]
y_cleaned = dum1[:,1:]

#print(x_cleaned.shape)
#print(y_cleaned.shape)

x_train = x_cleaned[:,5000:]
y_train = y_cleaned[:,5000:]

x_test = x_cleaned[:,:5000]
y_test = y_cleaned[:,:5000]

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

#y_train1 = y_train.T
w = np.zeros((200,16))
y_pred = np.matmul(x_train.T,w)
#print(y_pred.shape)
y_pred = np.exp(y_pred)
y_pred = np.matrix(y_pred)

sum_matrix = y_pred.sum(axis=1)
sum_matrix = np.matrix(sum_matrix)
#print(sum_matrix)
sum_matrix = sum_matrix.T
for r in range(5249):
    h = sum_matrix[0,r]
    y_pred[r,:] = y_pred[r,:]/h


#print(y_pred.shape)

prob  = y_pred

#y_pred_sum = sum(y_pred)

#y_prob = (y_pred)/y_pred_sum
y_pred = np.matmul(x_train.T, w)

y_pred = np.exp(y_pred)
y_pred = np.matrix(y_pred)

sum_matrix = y_pred.sum(axis=1)
sum_matrix = np.matrix(sum_matrix)

sum_matrix = sum_matrix.T
print('Initial probability matrix of training data generated')
for r in range(5249):
    h = sum_matrix[0, r]
    y_pred[r, :] = y_pred[r, :] / h
    prob = y_pred

C = np.zeros((5249, 16))
# print(C)
# Class_matrix = np.matrix(Class_matrix)


for s in range(5249):
    b = y_train[0, s]
    b = b.astype(int)
    C[s, b] = 1
    m = C - prob

gradient = np.matmul(x_train, m) * (1 / 5249)
# print(m)
# print(gradient)

for e in range(100):
    y_pred = np.matmul(x_train.T, w)

    y_pred = np.exp(y_pred)
    y_pred = np.matrix(y_pred)

    sum_matrix = y_pred.sum(axis=1)
    sum_matrix = np.matrix(sum_matrix)

    sum_matrix = sum_matrix.T
    for r in range(5249):
        h = sum_matrix[0, r]

        y_pred[r, :] = y_pred[r, :] / h
        prob = y_pred

# print(y_pred.shape)
print('Updating the weight matrix through Gradient Decent')

for s in range(5249):
    b = y_train[0, s]
    b = b.astype(int)
    C[s, b] = 1
    m = C - prob
w = w - 0.001 * gradient

# print(w)

sum_x = np.sum(x_test)
# normalization
x_test = (x_test) * 1 / (sum_x)
y_pred_test = np.matmul((x_test).T, w)

y_pred_test = np.exp(y_pred_test)
y_pred_test = np.matrix(y_pred_test)
sum_matrix1 = y_pred_test.sum(axis=1)
sum_matrix1 = np.matrix(sum_matrix1)
sum_matrix1 = sum_matrix1.T
for r1 in range(5000):
    h1 = sum_matrix1[0, r1]

    y_pred_test[r1, :] = 100 * y_pred_test[r1, :] / h1

# y_pred_test = np.floor(y_pred_test)

# print(max(y_pred_test(0)))

print('Testing the data')
y_pred_test = np.argmax(y_pred_test, axis=1)
#ensuring proper labeling
y_pred_test = (y_pred_test+1).T
# print(y_pred_test)
correct_mat = y_test - y_pred_test
correct_mat = correct_mat.T
correct = 0
for l in range(5000):
    if correct_mat[l] == 1:
        correct = correct + 1
    else:
        correct = correct

# % of Accuracy in prediction is =  (number of corrects/5000)*100


print('Accuracy of prediction = ', correct / 50)
