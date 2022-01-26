#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbor

# In[1]:


import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix


# In[2]:


df = pd.read_csv('real_309_4_en.csv')
df.head(3)


# In[4]:


#X = df.drop(columns = ['label', 'years', 'education', 'mind', 'financial', 'support'])
#y = df['label']

X = df.drop(columns = ['label', 'years', 'law'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 11)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

bestK = 0
bestP = 0
bestScore = 0

for k in range(1, 20):
    for p in range(1, 10):
        model = neighbors.KNeighborsClassifier(weights = 'distance', n_neighbors = k, p = p)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > bestScore:
            bestK = k
            bestP = p
            bestScore = score
            


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 11)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


model = neighbors.KNeighborsClassifier(algorithm = 'brute', n_neighbors = k, weights = 'distance', p = p)
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
con_matrix = confusion_matrix(y_test, y_pred)

print('number of correct sample: ', num_correct_samples)
print('accuracy: ', accuracy)
print('Precision:', precision_score(y_test, y_pred, average = 'weighted'))
print('Recall:', recall_score(y_test, y_pred, average = 'weighted'))
print('F1:', f1_score(y_test, y_pred, average = 'weighted'))
print('confusion matrix: ', con_matrix)


# In[5]:


plt.figure(figsize=(8,8))
sns.heatmap(con_matrix, annot=True)
plt.title('KNN \nAccuracy:{0:.2f}'.format(accuracy))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[6]:


from joblib import dump, load
dump(model, 'logistic_model.joblib') 


# In[ ]:





# In[ ]:




