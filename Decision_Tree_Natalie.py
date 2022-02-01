#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# train data

raw_data =  pd.read_csv("aug_train.csv")


# In[31]:


# preprocessing function
def preprocessing_data(df: pd.DataFrame):
    data = df.copy()
    # drop NaN values for some columns
    data = data.dropna(subset=['education_level','major_discipline', 'experience', 'last_new_job'])
    # Replace other NaN with Unknown value 
    data = data.replace(np.nan,'Unknown')
    # relevent_experience replace with 0 and 1, 1 for having experience and 0 for no experience
    data['relevent_experience'] = data['relevent_experience'].replace(['Has relevent experience','No relevent experience'],[1,0])

    # manually assign ordinal numbers to education_level and company_size
    # for graduate level I will give 1 and for master 2 and for phd 3. Graduate level can be equals to masters and phd but usually people with phd would not represent themselves as graduate. 
    # any graduate level certificate can be considered as graduate so I will assign a lower number to graduate than masters. 
    # for company_size unknown will get 0.
    
    data['education_level'] = data['education_level'].replace(['Graduate','Masters','Phd'],[1,2,3])
    data['company_size'] = data['company_size'].replace(['Unknown','<10', '10/49','50-99', '100-500','500-999','1000-4999','5000-9999','10000+'] ,range(0,9))

    # convert experience and last_new_job to numeric values
    data['experience'] = data['experience'].str.replace('>','').str.replace('<','')
    data['experience'] = pd.to_numeric(data['experience'])

    data['last_new_job'] = data['last_new_job'].str.replace('>','')
    data['last_new_job'] = data['last_new_job'].replace('never',0)
    data['last_new_job'] = pd.to_numeric(data['last_new_job'])

    data = pd.get_dummies(data, columns = ['company_type', 'enrolled_university', 'gender', 'major_discipline','city'])
    
    #Normalize data using MinMaxScaler function of sci-kit leaern
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_scaled = pd.DataFrame(x_scaled, columns = data.columns)
    return(data_scaled)


# In[32]:


processed_data = preprocessing_data(raw_data)
processed_data


# In[33]:


X = processed_data.drop(columns='target')
X


# In[34]:


Y = processed_data['target']
Y


# In[35]:


print(X.shape)
print(Y.shape)


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)


# In[38]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:





# In[39]:


origin_dt=DecisionTreeClassifier()


# In[40]:


origin_dt=origin_dt.fit(X_train,y_train)


# In[41]:


y_pred=origin_dt.predict(X_test)


# In[42]:


print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[43]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[ ]:





# In[ ]:





# In[44]:


gridsearch_dt=DecisionTreeClassifier(random_state=1)


# In[45]:


para_grid = {'criterion': ['gini','entropy'],
            'splitter' : ['best','random'],
            'max_depth' : [2,3,4,5,6,7,8],
            'min_samples_split' : [2,3,4,5],
            'min_samples_split' : [2,3,4,5]}


# In[46]:


gridsearch_dt = GridSearchCV(origin_dt,para_grid)


# In[47]:


gridsearch_dt.fit(X_train,y_train)


# In[48]:


gridsearch_dt.best_params_


# In[49]:


gridsearch_dt.best_estimator_


# In[50]:


grid_pred = gridsearch_dt.predict(X_test)


print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, grid_pred )))


# In[ ]:





# In[52]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, grid_pred)).plot();


# In[54]:


from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
from subprocess import call
from IPython.display import Image
estimator = gridsearch_dt.best_estimator_
dot_data = StringIO()
export_graphviz(estimator, out_file='tree.dot',  
                filled=True, rounded=True,
                special_characters=True,
                class_names = ['1','0'],
                feature_names = X_train.columns )
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename = 'tree.png')


# ## Randomly dropping feature to see the performance of model ( with parmeter criterion: 'gini',max_depth : 6, min_samples_split: 2, splitter: 'random'

# In[61]:


newdf = processed_data.drop(columns = 'education_level')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[62]:


processed_data.head()


# In[63]:


newdf = processed_data.drop(columns = 'relevent_experience')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[64]:


newdf = processed_data.drop(columns = 'city_development_index')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[65]:


newdf = processed_data.drop(columns = 'company_size')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[66]:


newdf = processed_data.drop(['last_new_job','education_level'], axis=1)
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[69]:


newdf = processed_data.drop(['experience','training_hours'], axis=1)
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
new_tree = DecisionTreeClassifier(criterion = 'gini',
 max_depth = 6,
 min_samples_split = 2,
 splitter= 'random',
 random_state = 1)
new_tree.fit(X_train, y_train)
y_pred = new_tree.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[ ]:




