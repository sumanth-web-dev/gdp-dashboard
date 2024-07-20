

# import streamlit as st
# import pandas as pd
# import math
# from pathlib import Path
# import os
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score


# # Set the title and favicon that appear in the Browser's tab bar.
# st.set_page_config(
#     page_title='GDP dashboard',
#     page_icon=':earth_americas:',  # This is an emoji shortcode.
# )

# DATA_FILENAME1 = 'data/my_data.csv'
# data = pd.read_csv(DATA_FILENAME1)

# st.subheader("50 Startup data")

# data["State"] = data["State"].astype('category').cat.codes

# x = data.iloc[:,:-1]
# y = data.iloc[:,1]
# lr = LinearRegression()
# lr.fit(x, y)  # Fit the model with the data
# predictions = lr.predict(x)  # Predict using the model
# r2 = r2_score(y, predictions)  # Calculate R^2 score

# st.subheader(f"R^2 score: {r2}")

# data





#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')




# In[2]:


@st.cache_data
def get_gdp_data():

    DATA_FILENAME = 'data/Churn_Modelling.csv'
    dataset = pd.read_csv(DATA_FILENAME)
    return dataset


dataset = get_gdp_data()


# In[3]:


st.write(dataset.head())


# In[4]:


st.write(dataset.info())


# # Check information of Dataset

# In[5]:


st.write(dataset.describe())


# In[6]:


st.write(dataset.isnull().sum())

# In[7]:


st.write(dataset.duplicated().sum())


# # Pre-Prunning

# In[8]:


dataset = dataset.iloc[:,3:]
st.write(dataset.head())


# # Encoding

# In[9]:


sns.countplot(x = dataset["Geography"])
st.pyplot(plt.gcf())




fig, ax = plt.subplots()
sns.barplot(dataset["Geography"])
st.pyplot(fig)


# In[10]:

sns.countplot(x = dataset["Gender"])
st.pyplot(plt.gcf())


# In[11]:


dataset = pd.get_dummies(dataset,  columns = ["Geography"] , drop_first= True , dtype=int)


# In[12]:


st.write(dataset)

# In[13]:


dataset = pd.get_dummies(dataset , columns = ["Gender"] , drop_first=True , dtype=int)


# In[14]:


st.write(dataset.head())


# In[15]:





plt.figure(figsize=(16,8))
sns.heatmap(dataset.corr() , annot = True , cmap='rainbow')
st.pyplot(plt.gcf())

# In[16]:


# In[17]:


x =  dataset.drop(['Exited'] , axis = 1)
y = dataset['Exited']


# In[18]:


st.write(y.value_counts())


# Imbalance Treatmeant required however we use stratify to balance training and test dataset

# In[19]:


import imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_smote , y_smote = smote.fit_resample(x,y)

st.write(y.value_counts())
st.write(y_smote.value_counts())


# In[ ]:





# In[20]:


from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x_smote ,y_smote , test_size= 0.2 , random_state= 33 , stratify= y_smote )


# In[21]:


# data Leakage Problem - 
## 1) if we jave train and test data separately, we jave to handle missing data, feature scaling, outlier treatment separetly

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[22]:


sns.barplot((y_train).value_counts())
st.pyplot(plt.gcf())

# In[23]:


def draw_histogram(dataset , varibale , n_rows , n_cols):
    fig = plt.figure(figsize=(16, 10))
    for i, var_name in enumerate(varibale):
        ax = fig.add_subplot(n_rows , n_cols , i+1)
        dataset[var_name].hist(bins=10 , ax = ax)
        ax.set_title(var_name + " Distribution")
    fig.tight_layout()
    st.pyplot(plt.gcf())
    
draw_histogram(dataset , x , 5 , 3)


# # Model Building 

# Model 1: ADABoost
# 
# Boosting is for handling high bias problem and bagging for high variance problem 

# In[24]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(x_train , y_train )

st.subheader("AdaBoostClassifier")

# In[25]:


y_predict_train_ada = ada.predict(x_train)
y_predict_test_ada = ada.predict(x_test)


# In[26]:


from sklearn import metrics

st.write(metrics.classification_report(y_train , y_predict_train_ada))
print()
st.write(metrics.classification_report(y_test , y_predict_test_ada))


# In[27]:


st.write(metrics.accuracy_score(y_train , y_predict_train_ada))
print()
st.write(metrics.accuracy_score(y_test , y_predict_test_ada))


# Model 2: Gradient Boosting Algorithm

# In[28]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train , y_train)


# In[29]:
st.subheader("GradientBoostingClassifier")

y_predict_train_gbc = gbc.predict(x_train)
y_predict_test_gbc = gbc.predict(x_test)


# In[30]:


st.write(metrics.classification_report(y_train , y_predict_train_gbc))
print()
st.write(metrics.classification_report(y_test , y_predict_test_gbc))


# In[31]:


st.write(metrics.accuracy_score(y_train , y_predict_train_gbc))
print()
st.write(metrics.accuracy_score(y_test , y_predict_test_gbc))


# Model 3: XGBoostClassificer

# In[32]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train , y_train )


# In[33]:
st.subheader("XGBClassifier")


y_predict_train_xgb = xgb.predict(x_train)
y_predict_test_xgb = xgb.predict(x_test)


# In[34]:


st.write(metrics.classification_report(y_train , y_predict_train_xgb))
print()
st.write(metrics.classification_report(y_test , y_predict_test_xgb))


# In[35]:


st.write(metrics.accuracy_score(y_train , y_predict_train_xgb))
print()
st.write(metrics.accuracy_score(y_test , y_predict_test_xgb))


# In[36]:

st.subheader('cross validation xgboost')
from sklearn.model_selection import cross_val_score
training_accuracy = cross_val_score(xgb , x_train , y_train , cv =10)
st.write(training_accuracy)
st.write(training_accuracy.mean())
st.write(training_accuracy.max())


# Model 4:Bagging Classifier

# In[37]:


from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier()
bagging.fit(x_train , y_train )


# In[38]:
st.subheader("BaggingClassifier")


y_predict_train_bagging = bagging.predict(x_train)
y_predict_test_bagging = bagging.predict(x_test)


# In[39]:


st.write(metrics.accuracy_score(y_train , y_predict_train_bagging))
print()
st.write(metrics.accuracy_score(y_test , y_predict_test_bagging))


# Model 5: RandomForestClassifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train , y_train )


# In[41]:
st.subheader("RandomForestClassifier")


y_predict_train_rfc = rfc.predict(x_train)
y_predict_test_rfc = rfc.predict(x_test)


# In[42]:


st.write(metrics.accuracy_score(y_train , y_predict_train_rfc))
print()
st.write(metrics.accuracy_score(y_test , y_predict_test_rfc))


# Model 6: KNN(K-Nearest Neighbors)

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(x_train , y_train )

st.subheader("KNeighborsClassifier")

# In[44]:


error_rate = []

for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train , y_train)
    y_pred = knn.predict(x_test)
    error_rate.append(np.mean(y_pred != y_test))


# In[45]:


error_rate


# In[46]:


plt.figure(figsize=(16,6))
plt.plot(range(1,15) , error_rate , color = 'red' ,linestyle = 'dashed' , marker = 'o' , markersize = 12 , markerfacecolor = 'blue')

plt.title("Error Rate vs K-value")
plt.xlabel('k-values')
plt.ylabel('Error Rate')
st.pyplot(plt.gcf())


# In[47]:


knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train , y_train )


# In[48]:


y_predict_train_knn= knn.predict(x_train)
y_predict_test_knn = knn.predict(x_test)


# In[49]:


print(metrics.accuracy_score(y_train , y_predict_train_knn))
print()
print(metrics.accuracy_score(y_test , y_predict_test_knn))


# # Voting classifier

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# In[51]:


c1 = LogisticRegression()
c2 = DecisionTreeClassifier()
c3 = RandomForestClassifier()
c4 = AdaBoostClassifier()
c5 = KNeighborsClassifier()
c6 = BaggingClassifier()
c7 = GradientBoostingClassifier()
c8 = XGBClassifier()


# In[52]:


estimators = [('lr', c1) ,('Dtc' , c2) , ('rfc' , c3) ,('Ada' , c4 ) , ('Knn' , c5),
             ('bagging' , c6) , ('gr' , c7) , ('xg' , c8) ]


# In[61]:


for estimator in estimators:
    x = cross_val_score(estimator[1] , x_smote , y_smote , cv = 10 , scoring='accuracy')
    st.write(estimator[0] , np.round(np.mean(x) , 2))


# In[62]:

st.subheader("VotingClassifier")

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators)

vc_cross_val = cross_val_score(vc , x_smote , y_smote ,cv = 10,  scoring='accuracy')


st.write(label = 'cross val mean', value = vc_cross_val.mean())

st.write(label = 'cross_val_max' , value = vc_cross_val.max())


# In[ ]:



# # Stacking 

# In[63]:


from sklearn.ensemble import StackingClassifier

sc = StackingClassifier(estimators , final_estimator=c1)

st.subheader("StackingClassifier")

vc_cross_val1 = cross_val_score(vc , x_smote , y_smote ,cv = 10,  scoring='accuracy')

st.write(vc_cross_val1)
st.write(vc_cross_val1.mean())

st.write(print(vc_cross_val1.max()))


# In[ ]:




