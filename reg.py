
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import os
os.chdir('c://users/sujith/desktop')
data=pd.read_csv('iris.csv')
data.head()


# In[2]:


data.tail()


# In[3]:


data.describe()


# In[4]:


data.mean()


# In[5]:


data.isnull()


# In[6]:


data.shape


# In[7]:


data.sepal_length.describe()


# In[8]:


data.sepal_length.unique()


# In[9]:


import seaborn as sns
sns.pairplot(data, x_vars=['sepal_length','sepal_width'], y_vars=['petal_length','petal_width'],size=7,aspect=0.7,kind='reg')


# In[10]:


feature_cols=['sepal_length','sepal_width']
x=data[feature_cols]
x.head()


# In[11]:


print(type(x))
print(x.shape)


# In[12]:


feature_cols=['petal_length','petal_width']
y=data[feature_cols]
y.head()


# In[13]:


print(type(y))
print(y.shape)


# In[14]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)


# In[15]:


print(x_train.shape)
print(y_train.shape)
print(x_train.shape)
print(y_train.shape)


# In[16]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[17]:


linreg.fit(x_train,y_train)


# In[18]:


print(linreg.intercept_)
print(linreg.coef_)


# In[19]:


list(zip(feature_cols,linreg.coef_))


# In[20]:


y_pred=linreg.predict(x_test)


# In[21]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


