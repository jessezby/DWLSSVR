
from numpy import *
import numpy as np
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append(float(lineArr[0]))
        labelMat.append(float(lineArr[1]))
    return mat(dataMat).T,mat(labelMat).T
           

def kernelTrans(X,A,kTup):
    X = mat(X)
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem ,That Kernel is not recognized')
    return K
    
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.K = mat(zeros((self.m,self.m)))  
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
            

def leastSquares(dataMatIn,classLabels,C,kTup):
  
    oS = optStruct(dataMatIn,classLabels,C,kTup)
    unit = mat(ones((oS.m,1)))  
    I = eye(oS.m)
    zero = mat(zeros((1,1)))
    upmat = hstack((zero,unit.T))
    downmat = hstack((unit,oS.K + I/float(C)))
    completemat = vstack((upmat,downmat))  
    rightmat = vstack((zero,oS.labelMat))  
    b_alpha = completemat.I * rightmat 
    oS.b = b_alpha[0,0]
    for i in range(oS.m):
        oS.alphas[i,0] = b_alpha[i+1,0]
    e = oS.alphas/C   
    return oS.alphas,oS.b,e

def weights(e):
    c1 = 2.5
    c2 = 3
    m = shape(e)[0]
    v = mat(zeros((m,1)))
    v1 = eye(m)
    q1 = int(m/4.0)
    q3 = int((m*3.0)/4.0)
    e1 = []
    shang = mat(zeros((m,1)))
    for i in range(m):
        e1.append(e[i,0])
    e1.sort()
    IQR = e1[q3] - e1[q1]
    s = IQR/(2 * 0.6745)
    for j in range(m):
        shang[j,0] = abs(e[j,0]/s)
    for x in range(m):
        if shang[x,0] <= c1:
            v[x,0] = 1.0
        if shang[x,0] > c1 and shang[x,0] <= c2:
            v[x,0] = (c2 - shang[x,0])/(c2 - c1)
        if shang[x,0] > c2:
            v[x,0] = 0.0001
        v1[x,x] = 1/float(v[x,0])
    return v1

def weightsleastSquares(dataMatIn,classLabels,C,kTup,v1):
    oS = optStruct(dataMatIn,classLabels,C,kTup)
    unit = mat(ones((oS.m,1))) 
    gamma = kTup[1]
    zero = mat(zeros((1,1)))
    upmat = hstack((zero,unit.T))
    downmat = hstack((unit,oS.K + v1/float(C)))
    completemat = vstack((upmat,downmat))  
    rightmat = vstack((zero,oS.labelMat)) 
    b_alpha = completemat.I * rightmat
    oS.b = b_alpha[0,0]
    for i in range(oS.m):
        oS.alphas[i,0] = b_alpha[i+1,0]
    e = oS.alphas/C
    return oS.alphas,oS.b


def predict(alphas,b,dataMat):
    m,n = shape(dataMat)
    predict_result = mat(zeros((m,1)))
    for i in range(m):
        Kx = kernelTrans(dataMat,dataMat[i,:],kTup)  
        predict_result[i,0] =  Kx.T * alphas + b   
    return predict_result

def predict_average_error(predict_result,label):
    m,n = shape(predict_result)
    error = 0.0
    for i in range(m):
        error += abs(predict_result[i,0] - label[i,0])
    average_error = error / m
    return average_error
    


if __name__ == '__main__':
    print('--------------------Load Data------------------------')
    dataMat,labelMat = loadDataSet('sine.txt')
    print('--------------------Parameter Setup------------------')
    C = 0.6
    k1 = 0.3
    kernel = 'rbf'
    kTup = (kernel,k1)
    print('-------------------Save LSSVM Model-----------------')
    alphas,b,e = leastSquares(dataMat,labelMat,C,kTup)
    print('----------------Calculate Error Weights-------------')
    v1 = weights(e)
    print('------------------Save WLSSVM Model--------------- -')
    alphas1,b1 = weightsleastSquares(dataMat,labelMat,C,kTup,v1)
    print('------------------Predict Result------------------ -')
    predict_result = predict(alphas1,b1,dataMat)
    print('-------------------Average Error------------------ -')
    average_error = predict_average_error(predict_result,labelMat)
    print(average_error)
    





