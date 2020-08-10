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

from sklearn.ensemble import RandomForestClassifier
R=RandomForestClassifier(n_estimators=55)
R.fit(X_train,Y_train)

Y_pred_R=R.predict(X_test)

from sklearn.metrics import accuracy_score
R_accuracy=accuracy_score(Y_test,Y_pred_R)
print("The accuracy of the RandomForestClassifier model is :",(R_accuracy)*100,'%')






