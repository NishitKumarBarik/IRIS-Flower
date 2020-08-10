#load iris dataset to our program
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data    #contains data
Y=iris.target  #contains results

#We analyze and describe the features and targets of iris dataset
print('The input data is :',X)     #will display all features of iris dataset
print('The output data is :',Y)    #will display all the target of iris data set(result)
print('The size of the input data is :',X.shape)
print('The size of the output data is :',Y.shape)

#The dataset was splitted. (80% for training and 20% for testing). This was done by importing #train_test_split function .
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print('The size of the input data to be trained is :',X_train.shape)
print('The size of the output data to be trained is :',Y_train.shape)

#In this step we use k-nearest neighbors algorithm to train the 80% dataset .
#Here value of k is 5
from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier(n_neighbors=5)
K.fit(X_train,Y_train)

Y_pred_K=K.predict(X_test)
#Here we tested the 30% of data set to predict the targets

from sklearn.metrics import accuracy_score
K_accuracy=accuracy_score(Y_test,Y_pred_K)
print("The accuracy of the kNeighboursClassifier model is :",(K_accuracy)*100,'%')
