from sklearn.datasets import load_iris
import tkinter.messagebox as m
iris=load_iris()
x=iris.data
y=iris.target


##########split data##########
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


##########implement RandomForestClassifier##########
def rfc():
    global acc_rfc
    global R
    from sklearn.ensemble import RandomForestClassifier
    R=RandomForestClassifier(n_estimators=55)
##########train the model##########
    R.fit(x_train,y_train)
##########test the model##########
    y_pred_rfc=R.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_rfc=accuracy_score(y_test,y_pred_rfc)
    acc_rfc=round(acc_rfc*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of RandomForestClassifier model is : "+str(acc_rfc))
    
    
##########implement supportvectorMachine##########
def svm():
    global acc_svm
    global S
    from sklearn.svm import SVC
    S=SVC(gamma='auto')
##########train the model##########
    S.fit(x_train,y_train)
##########test the model##########
    y_pred_svm=S.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_svm=accuracy_score(y_test,y_pred_svm)
    acc_svm=round(acc_svm*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of supportvectorMachine model is : "+str(acc_svm))
    
    
##########implement KNNeighbors##########
def knn():
    global acc_knn
    global K
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
##########train the model##########
    K.fit(x_train,y_train)
##########test the model##########
    y_pred_knn=K.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_knn=accuracy_score(y_test,y_pred_knn)
    acc_knn=round(acc_knn*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of KNNeighbors model is : "+str(acc_knn))
    
    
##########Implement LogsticRegression##########
def lr():
    global acc_lr
    global L
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression(solver='liblinear',multi_class='auto')
##########train the model##########
    L.fit(x_train,y_train)
##########test model##########
    y_pred_lr=L.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_lr=accuracy_score(y_test,y_pred_lr)
    acc_lr=round(acc_lr*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of LogisticRegression model is : "+str(acc_lr))
    
    
##########Implement DecisionTree##########
def dt():
    global acc_dt
    global D    
    from sklearn.tree import DecisionTreeClassifier
    D=DecisionTreeClassifier()
##########train the model##########
    D.fit(x_train,y_train)
##########test model##########
    y_pred_dt=D.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(y_test,y_pred_dt)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of DecisionTree model is : "+str(acc_dt))
    
    
##########Implement NaiveBayes##########
def nb():
    global acc_nb
    global N
    from sklearn.naive_bayes import GaussianNB
    N=GaussianNB()
##########train the model##########
    N.fit(x_train,y_train)
##########test model##########
    y_pred_nb=N.predict(x_test)
##########Find accuracy##########
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(y_test,y_pred_nb)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title="IRISspecies",message="The accuracy of NaiveBayes model is : "+str(acc_nb))
    
    
####################
def compare():
    import matplotlib.pyplot as plt
    global acc_rfc
    global acc_svm
    global acc_knn
    global acc_lr
    global acc_dt
    global acc_nb
    model=['LG','SVM','KNN',"RFC",'DT','NB']
    accuracy=[acc_lr,acc_svm,acc_knn,acc_rfc,acc_dt,acc_nb]
    plt.bar(model,accuracy,color=['yellow','red','blue','black','magenta','green'])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
    
    
####################
def predict():
    global R
    global S
    global K
    global D
    global L
    global N
    a=int(v1.get())
    b=int(v2.get())
    c=int(v3.get())
    d=int(v4.get())
    data1={acc_rfc:R.predict([[a,b,c,d]]),acc_svm:S.predict([[a,b,c,d]]),acc_knn:K.predict([[a,b,c,d]]),acc_lr:L.predict([[a,b,c,d]]),acc_dt:D.predict([[a,b,c,d]]),acc_nb:N.predict([[a,b,c,d]])}
    data2={acc_rfc:'RandomForestClassifier',acc_svm:'supportvectorMachine',acc_knn:'KNNeighbors',acc_lr:'LogisticRegression',acc_dt:'DecisionTree',acc_nb:'NaiveBayes'}
    result=data1.get(max(acc_rfc,acc_svm,acc_knn,acc_lr,acc_dt,acc_nb))
    model=data2.get(max(acc_rfc,acc_svm,acc_knn,acc_lr,acc_dt,acc_nb))
    if result==0:
        m.showinfo(title="IRIS",message="The most accurate model is "+model+" and the flower is IRIS SETOSA")
    elif result==1:
        m.showinfo(title="IRIS",message="The most accurate model is "+model+" and the flower is IRIS VERSICOLOR")
    else:
        m.showinfo(title="IRIS",message="The most accurate model is "+model+" and the flower is IRIS VIRGINICA")
        
        
####################
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")
    
    
##########DESIGN##########
from tkinter import *
w=Tk()
w.title("IRIS FLOWER SPECIES PREDICTOR")
w.configure(bg='cyan')
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
Brfc=Button(w,text="RFC",bg='black',fg='white',command=rfc)
Bsvm=Button(w,text="SVM",bg='red',fg='white',command=svm)
Bknn=Button(w,text="KNN",bg='blue',fg='white',command=knn)
Blr=Button(w,text="LR",bg='yellow',command=lr)
Bdt=Button(w,text="DT",bg='magenta',fg='white',command=dt)
Bnb=Button(w,text="NB",bg='green',fg='white',command=nb)
Bcmp=Button(w,text="                  COMPARE               ",bg='magenta',fg='white',command=compare)
Bsubmit=Button(w,text="SUBMIT",bg='green',fg='white',command=predict)
Breset=Button(w,text="RESET",bg='red',fg='white',command=reset)
####################
L1=Label(w,text="Enter the Sepal length",bg='cyan')
L2=Label(w,text="Enter the Sepal width",bg='cyan')
L3=Label(w,text="Enter the Petal length",bg='cyan')
L4=Label(w,text="Enter the Petal width",bg='cyan')
L5=Label(w,text="Click on a model to know its accuracy",bg='cyan')
L6=Label(w,text="Click here to compare the above models",bg='cyan')
####################
'''img=PhotoImage(file='D:\PYTHON Programing\Iris Species Predictor\irisimg.GIF')
lblimg=Label(w,image=img,width=175,height=70)
lblimg.grid(row=4,column=4,columnspan=3,rowspan=4)'''
####################
E1=Entry(w,textvariable=v1)
E2=Entry(w,textvariable=v2)
E3=Entry(w,textvariable=v3)
E4=Entry(w,textvariable=v4)
####################
Brfc.grid(row=2,column=1)
Bdt.grid(row=2,column=2)
Bsvm.grid(row=2,column=3)
Bnb.grid(row=2,column=4)
Bknn.grid(row=2,column=5)
Blr.grid(row=2,column=6)
Bcmp.grid(row=3,column=5,columnspan=2)
Bsubmit.grid(row=8,column=3)
Breset.grid(row=8,column=5,columnspan=2)
####################
L1.grid(row=4,column=1,columnspan=2)
E1.grid(row=4,column=3)
####################
L2.grid(row=5,column=1,columnspan=2)
E2.grid(row=5,column=3)
####################
L3.grid(row=6,column=1,columnspan=2)
E3.grid(row=6,column=3)
####################
L4.grid(row=7,column=1,columnspan=2)
E4.grid(row=7,column=3)
####################
L5.grid(row=1,column=1,columnspan=6)
L6.grid(row=3,column=1,columnspan=4)
####################
w.mainloop()
