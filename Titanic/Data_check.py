#coding utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv').replace("male",0).replace("female",1)
columns = ['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data = df[columns]
#df = df.drop(["Embarked","Sex","Name","Cabin"],axis=1)
df["Age"].fillna(df.Age.median(), inplace=True)     #欠損値を中央値に
#SEX = df['Sex'].map({"female":0,"male":1}).astype(int) #female = 0 , male = 1
#Emb = df['Embarked'].map({"S":0,"Q":1,"C":1}).astype(int)

Emb_df = pd.get_dummies(df[['Embarked']])    #PandasでOne-Hot形式に変換
Cabin_df = pd.get_dummies(df[['Cabin']])    #PandasでOne-Hot形式に変換
Ticket_df = pd.get_dummies(df[['Ticket']])
data = pd.concat([Cabin_df,Ticket_df,Emb_df,df],axis = 1)
data = data.drop(['Cabin','Embarked','Ticket','Name'],axis = 1)

X_var = data.drop('Survived', axis=1)
X_array = X_var.as_matrix()

y_var = data['Survived']
y_array = y_var.as_matrix()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2,random_state=0)

from sklearn import linear_model
model=linear_model.LinearRegression(normalize=True)
model.fit(X_train, y_train)

print(model.score(X_train, y_train),"%")
print(model.score(X_test, y_test),"%")
##ひｄっどおおおおおい