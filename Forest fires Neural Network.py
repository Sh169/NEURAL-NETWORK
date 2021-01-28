#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


# In[65]:


#Load the data
forestfires = pd.read_csv("forestfires.csv")


# In[66]:


forestfires.head()


# In[67]:


#To understand the datatypes
forestfires.dtypes


# In[68]:


#Drop the columns month and day
forestfires = forestfires.drop(['month'], axis=1)
forestfires = forestfires.drop(['day'], axis=1)
forestfires.head()


# In[69]:


# Dataset Categorical variables encoding
forestfires['size_category'] = forestfires['size_category'].map({'small':0, 'large':1})


# In[70]:


#Correlation matrix
sns.heatmap(forestfires.corr()>0.6, cmap='Greens')


# In[71]:


#Checking for the null attributes
forestfires.isnull().sum()


# In[72]:


forestfires.describe()


# In[73]:


# Boxplot
sns.boxplot(data = forestfires, orient = "h")


# In[74]:


sns.countplot(forestfires['size_category'])
plt.show()


# In[75]:


# Splitting dataset
X = forestfires.drop(['size_category'], axis=1)
y = forestfires['size_category']


# In[76]:


# Transformation MinMaxScalar
from sklearn.preprocessing import MinMaxScaler


# In[77]:


scaler = MinMaxScaler()

X = scaler.fit_transform(X)


# In[78]:


# Splitting data into train & test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.2, random_state=42 )


# In[79]:


((X_train.shape, y_train.shape),(X_test.shape, y_test.shape))


# In[80]:


# # Neural Network Model
# generating the data set
from sklearn.datasets import make_classification
X, y = make_classification(n_features =2, n_redundant =0, n_informative=2, random_state=3)


# In[81]:


# Visualization
plt.scatter(X[y==0][:,0], X[y==0][:,1], s=100, edgecolors='k')
plt.scatter(X[y==1][:,0], X[y==1][:,1], s=100, edgecolors='k', marker='^')
plt.show()


# In[82]:


# MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


# ###### Getting mlp score

# In[84]:


from sklearn.neural_network import MLPClassifier
X, y = make_classification(n_features =2, n_redundant=0, n_informative=2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))


# In[85]:


mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50))


# In[86]:


from sklearn.neural_network import MLPClassifier

#increasing the hidden layers gives more accuracy
clf = MLPClassifier(activation ='relu',solver='lbfgs', alpha=0.0001,hidden_layer_sizes=(3), random_state=1)
clf.fit(X,y)


# In[87]:


pred_values = clf.predict(X)
print(pred_values)


# In[88]:


import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[89]:


confusion_matrix = confusion_matrix(y,pred_values)
confusion_matrix


# In[90]:


classification_report = classification_report(y,pred_values)
print(classification_report)


# In[91]:


print("Accuracy:",metrics.accuracy_score(y,pred_values))


# ###### This model we have accuracy of 87%
