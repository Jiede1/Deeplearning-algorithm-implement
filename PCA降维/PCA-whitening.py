#implement PCA
file=open('/notebooks/pcaData.txt','r')
dataSet=[]
for text in file:
    tt=text.strip().split()
    line=[]
    for t in tt:
        line.append(float(t))
    dataSet.append(line)
dataSet=np.array(dataSet)
dataSet.shape  #(2,45)

import matplotlib.pylab as plt
%matplotlib inline

#画出原数据
plt.figure(1)
plt.scatter(dataSet[0,:],dataSet[1,:])
plt.title("origin data")

#计算协方差矩阵sigma，以及特征向量矩阵u
sigma=dataSet.dot(dataSet.T)/dataSet.shape[1]
print(sigma.shape)  #(2,2)
[u,s,v] = np.linalg.svd(sigma)
print(u.shape)  #(2,2)

#画出两个主成分方向
plt.figure(2)
plt.plot([0, u[0,0]], [0, u[1,0]])   
plt.plot([0, u[0,1]], [0, u[1,1]])
plt.scatter(dataSet[0,:],dataSet[1,:])

#PCA转换数据,不降维
xRot=u.T.dot(dataSet)
xRot.shape   #(2,45)

#画出PCA转换后的数据
plt.figure(3)
plt.scatter(xRot[0,:], xRot[1,:])
plt.title('xRot')

k = 1; #降维度为1

#PCA降维，xRot[0:k,:] 为降维度后的数据
xRot[0:k,:] = u[:,0:k].T .dot(dataSet) 
#还原数据
xHat = u .dot(xRot)
print(xHat.shape)
plt.figure(4)
plt.scatter(xHat[0,:], xHat[1, :])
plt.title('xHat')

#PCA Whitening
# Complute xPCAWhite and plot the results.

epsilon = 1e-5

#这部分用到了技巧，利用s的元素运算后（防止数据不稳定或数据溢大，具体看原理），再恢复对角矩阵。具体见diag函数
xPCAWhite = np.diag(1./np.sqrt(s + epsilon)) .dot(u.T .dot(dataSet)) 

plt.figure(5)
plt.scatter(xPCAWhite[0, :], xPCAWhite[1, :])
plt.title('xPCAWhite')

#ZCA白化
xZCAWhite = u .dot(np.diag(1./np.sqrt(s + epsilon)))  .dot(u.T .dot(dataSet)) 


plt.figure(6)
plt.scatter(xZCAWhite[0, :], xZCAWhite[1, :])
plt.title('xZCAWhite')
plt.show()
