from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from scipy.optimize import minimize

data=load_iris()
dataSet=data['data']
classLabels=data['target']
m,n=dataSet.shape
k=len(np.unique(classLabels))
#打乱数据
listt=shuffle(np.arange(dataSet.shape[0]))
dataSet=dataSet[listt]
classLabels=classLabels[listt]

def sigmoid(X):
    return 1/(1+np.exp(-X))

#theta.shape==(k,n+1)
def J(X,classLabels,theta,alpha,lenda): 
    bin_classLabels=label_binarize(classLabels,classes=np.unique(classLabels).tolist()).reshape((m,k))  #二值化 (m*k) 
    dataSet=np.concatenate((X,np.ones((m,1))),axis=1).reshape((m,n+1)).T   #转换为（n+1,m）
    theta_data=theta.dot(dataSet)  #(k,m)
    theta_data = theta_data - np.max(theta_data)   #k*m
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)  #(k*m)
    #print(bin_classLabels.shape,prob_data.shape
    cost = (-1 / m) * np.sum(np.multiply(bin_classLabels,np.log(prob_data).T)) + (lenda / 2) * np.sum(np.square(theta))  #标量
    #print(dataSet.shape,prob_data.shape)
    grad = (-1 / m) * (dataSet.dot(bin_classLabels - prob_data.T)).T + lenda * theta  #(k*N+1)

    return cost,grad

def train(X,classLabels,theta,alpha=0.1,lenda=1e-4,maxiter=1000):
    #options_ = {'maxiter': 400, 'disp': True}
    #result =minimize(J(X,classLabels,theta,alpha,lenda), theta, method='L-BFGS-B', jac=True, options=options_)
    #return result.x
    for i in range(maxiter):
        cost,grad=J(X,classLabels,theta,alpha,lenda)
        theta=theta-alpha*grad
    return theta

def predict(theta,testSet,testClass):  #testSet (m,n+1)
    prod = theta.dot(testSet.T)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)
    accuracy=0.0
    for i in range(len(testClass)):
        if testClass[i]==pred[i]:
            accuracy+=1

    return pred,float(accuracy/len(testClass))

def check_gradient(X,classLabels,theta,alpha,lenda,eplison=1e-4):    # gradient check
    cost= lambda theta:J(X,classLabels,theta,alpha,lenda)
    print("Norm of the difference between numerical and analytical num_grad (should be < 1e-9)\n")
    print(theta.shape)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            #print(i,j)
            theta_1=np.array(theta)
            theta_2=np.array(theta)
            theta_1[i,j]=theta[i,j]+eplison
            theta_2[i,j]=theta[i,j]-eplison
            num_cost_1,x=cost(theta_1)
            num_cost_2,x=cost(theta_2)
            num_grad=(num_cost_1-num_cost_2)/(2*eplison)
            x,grad=cost(theta)
            iff = np.linalg.norm(num_grad- grad[i,j]) / np.linalg.norm(num_grad + grad[i,j])
            print("the difference of the grad and num_grad: ",iff)
            #print("Norm of the difference between numerical and analytical num_grad (should be < 1e-7)\n")

theta=np.random.random((k,n+1))   #初始化theta
check_gradient(dataSet,classLabels,theta,alpha=0.1,lenda=1e-4,eplison=1e-4)  #首先进行梯度检验
theta_train=train(dataSet,classLabels,theta,alpha=0.1)    #训练数据,在这里我不太严格，将所有数据都用于训练了
testSet=np.concatenate((dataSet,np.ones((m,1))),axis=1).reshape((m,n+1))    #所有训练数据同时作为预测数据，不建议，我这只是为了方便

pred,accuracy=predict(theta_train,testSet,classLabels)  #预测
print("accuracy: ",accuracy)   #最终准确率在98%左右
