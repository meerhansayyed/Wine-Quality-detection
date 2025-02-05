#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('winequality.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.isnull().sum()


# # DATA ANALYSIS AND VISUALIZATION

# In[8]:


# statistical measure of the dataset
data.describe()


# In[9]:


# number of values for each quality 
sns.catplot(x='quality', data = data, kind = 'count')


# In[10]:


# volatile acidity vs quality
plot = plt.figure(figsize = (5,5))
sns.barplot(x='quality',y='volatile acidity', data=data)


# In[11]:


#compare citric acid value and quality 
plot = plt.figure(figsize = (5,5))
sns.barplot(x='quality',y='citric acid', data=data)


# # correlation to  find wine quality
# 

# 1. Positive Correlation
# 2. Negative Correlation

# In[12]:


correlation = data.corr()


# In[13]:


#constructing heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt= '.1f', annot=True, annot_kws={'size':8},cmap='Blues')


# # Data Preprocessing 

# In[14]:


# separate the data and label
x = data.drop('quality',axis=1)


# In[15]:


print(x)


# # label Binarization

# In[16]:


#instead of having more values i need only 2 values which is good or bat this is binarization
y = data['quality'].apply(lambda y_value : 1 if y_value  >=7 else 0)#lambda funtion is a function which help to chage the values in python


# In[17]:


print(y)


# # Spliting the data into training and test data 

# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=3)


# In[19]:


print(y.shape , y_train.shape , y_test.shape)


# # Training the model using random forestClassifier 
# #random forest is a combination of multiple Decision Tree, more the number of decision tree better the prediction
# 
# 
# 

# In[21]:


model = RandomForestClassifier()


# In[23]:


model.fit(x_train,y_train)


# Model evaluation
# accurracy Score

# In[26]:


#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[29]:


print("Accuracy :" , test_data_accuracy)


# # Building a Predictive System

# In[38]:


input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

#changing the input data to a numpy array

input_data_as_numpy_array = np.array(input_data)

#reshape the data as we are predictingthe label for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('it is a good quality wine')
else:
    print('it is a bad quality wine ')


# In[ ]:




