#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import pickle 
import seaborn as sns


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
df


# In[3]:


df.describe() #no null value


# In[4]:


ProfileReport(df)


# In[5]:


df.columns


# In[6]:


df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())


# In[7]:


ProfileReport(df)


# In[8]:


#Handling Outliers
fig,ax=plt.subplots(figsize= (20,20))
sns.boxplot(data=df,ax=ax)


# In[9]:


df.columns


# In[10]:


q=df['Insulin'].quantile(0.70)


# In[11]:


q


# In[12]:


df_new=df[df['Insulin'] <q ]


# In[13]:


ProfileReport(df_new)


# In[14]:


df


# In[15]:


df_new #We are loosing data as a result of handling outliers


# In[16]:


fig,ax=plt.subplots(figsize= (20,20))
sns.boxplot(data=df_new,ax=ax)


# In[17]:


df.columns


# In[18]:


q=df['Pregnancies'].quantile(.98)
df_new=df[df['Pregnancies']<q] #removes 2% of data outliers


# In[19]:


q=df_new['BMI'].quantile(.99) #want o keep only 99 pct of dataset
df_new=df_new[df_new['BMI']<q]


# In[20]:


q=df_new['Insulin'].quantile(.99) #want o keep only 99 pct of dataset
df_new=df_new[df_new['Insulin']<q]#you need to do tries in order gto get the number here like .99


# In[21]:


q=df_new['DiabetesPedigreeFunction'].quantile(.99) #want o keep only 99 pct of dataset
df_new=df_new[df_new['DiabetesPedigreeFunction']<q]


# In[22]:


q=df_new['Age'].quantile(.99) #want o keep only 99 pct of dataset
df_new=df_new[df_new['Age']<q]


# In[23]:


df_new


# In[24]:


y=df_new.Outcome


# In[25]:


x=df_new.drop(columns=['Outcome'])


# In[26]:


scalar=StandardScaler()


# In[27]:


x_scaled=scalar.fit_transform(x)


# In[28]:


x_scaled


# In[29]:


ProfileReport(pd.DataFrame(x_scaled))


# In[30]:


fig,ax=plt.subplots(figsize= (20,20))
sns.boxplot(data=pd.DataFrame(x_scaled),ax=ax)


# In[31]:


def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=["FEATURE", "VIF_SCORE"])


# In[32]:


vif_score(x)


# In[33]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=.20,random_state=0)


# In[34]:


logi=LogisticRegression(verbose=1,n_jobs=4,solver='liblinear')#For small datasets liblinear is a good choice


# In[35]:


logi.fit(x_train,y_train) 


# In[36]:


x_test[0]


# In[37]:


df


# In[48]:


y_pred  = logi.predict(x_test)


# In[52]:


logi.predict(scalar.fit_transform([[6,148.0,72.0,35.000000,79.799479,33.6,0.627,50]]))          


# In[53]:


scalar.transform([[6,148.0,72.0,35.000000,79.799479,33.6,0.627,50]])


# In[39]:


logi.predict([x_test[54]])


# In[40]:


logi.predict([x_test[82]])


# In[41]:


y_pred=logi.predict(x_test)


# # Drawing confusion matrix

# In[43]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve 


# In[44]:


confusion_matrix(y_test,y_pred) #from the output we can evaluate accuracy, precision, recall...


# In[45]:


confusion_matrix(y_test,y_pred).ravel() #tn fp fn tp


# In[54]:


tn , fp , fn , tp = confusion_matrix(y_test,y_pred).ravel()


# In[57]:


fn


# In[59]:


accuracy=(tp+tn)/(tp+tn+fp+fn)
accuracy


# In[60]:


precision=tp/(tp+fp)
precision


# In[61]:


recall=tp/(tp+fn)
recall


# In[62]:


F1_Score = 2*(recall * precision) / (recall + precision)
F1_Score


# In[ ]:




