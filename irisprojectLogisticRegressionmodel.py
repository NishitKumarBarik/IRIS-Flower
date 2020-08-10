from sklearn.datasets import load_iris
iris = load_iris()
X=iris.data
Y=iris.target

print('The input data is :',X)  
print('The output data is :',Y)  
print('The size of the input data is :',X.shape)
print('The size of the output data is :',Y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
print('The size of the input data to be trained is :',X_train.shape)
print('The size of the output data to be trained is :',Y_train.shape)

from sklearn.linear_model import LogisticRegression
L=LogisticRegression(multi_class='auto',solver='liblinear')
L.fit(X_train,Y_train)

Y_pred_L=L.predict(X_test)

from sklearn.metrics import accuracy_score
L_accuracy=accuracy_score(Y_test,Y_pred_L)
print("The accuracy of the LogisticRegression model is :",(L_accuracy)*100,'%')



