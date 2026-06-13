#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,accuracy_score


# In[4]:


data=pd.read_csv(r"C:\Users\flypbox\Downloads\archive (4)\heart.csv")
data.head(10)


# In[16]:


feature_cols=['chol','fbs','thal','trestbps','thalach','ca','exang']
x=data[feature_cols]
y=data.iloc[:,1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[17]:


# Used SVM classifier
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy : ", accuracy) 
print("-"*60)
print("classification_report : \n",classification_report(y_pred,y_test)) 
print("-"*60)
print("confuison_matrix : \n",confusion_matrix(y_test,y_pred))
print("-"*60)
print("roc_auc_score : ",roc_auc_score(y_pred,y_test))


# In[18]:


# Used random forest classifier 
model=RandomForestClassifier(9)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy : ", accuracy) 
print("-"*60)
print("classification_report : \n",classification_report(y_pred,y_test)) 
print("-"*60)
print("confuison_matrix : \n",confusion_matrix(y_test,y_pred))
print("-"*60)
print("roc_auc_score : ",roc_auc_score(y_pred,y_test))


# In[19]:


# Used Logistic regression 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=500)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy : ", accuracy) 
print("-"*60)
print("classification_report : \n",classification_report(y_pred,y_test)) 
print("-"*60)
print("confuison_matrix : \n",confusion_matrix(y_test,y_pred))
print("-"*60)
print("roc_auc_score : ",roc_auc_score(y_pred,y_test))


# 
