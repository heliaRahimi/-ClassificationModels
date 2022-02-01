#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
# train data

raw_data =  pd.read_csv("aug_train.csv")


# In[3]:


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


# In[4]:


processed_data = preprocessing_data(raw_data)
processed_data


# In[5]:


X = processed_data.drop(columns='target')
X


# In[6]:


Y = processed_data['target']
Y


# In[7]:


print(X.shape)
print(Y.shape)


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)


# In[9]:


print(X_train.shape)
print(y_train.shape)


# In[10]:


rf1 = RandomForestClassifier()


# In[11]:


rf1 = rf1.fit(X_train,y_train)


# In[12]:


y_pred = rf1.predict(X_test)


# In[13]:


print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[14]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# ## Use GridSearchCV to find a better combination of parameter for random forest

# In[19]:


para_grid = {'n_estimators': [10,50,100,150],
            'max_depth' : [3,6,9],
            'max_features' : ['auto', 'sqrt', 0.5,0.6,0.7,0.8,0.9],
            'max_leaf_nodes' : [10,15,20,25],
            'min_samples_split' : [2,5,10],
            'bootstrap' : [True,False]}


# In[20]:


rf2 = RandomForestClassifier(random_state=1)


# In[21]:


r_search = GridSearchCV(rf2,para_grid,cv=3,scoring='roc_auc')


# In[22]:


r_search.fit(X_train,y_train)


# In[23]:


r_search.best_params_


# In[29]:


r_search.best_estimator_


# In[30]:


grid_pred = r_search.predict(X_test)


print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, grid_pred)))


# In[31]:


n_nodes = []
max_depths = []

for ind_tree in r_search.best_estimator_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')


# In[32]:


train_rf_predictions = r_search.predict(X_train)
train_rf_probs = r_search.predict_proba(X_train)[:, 1]

rf_predictions = r_search.predict(X_test)
rf_probs = r_search.predict_proba(X_test)[:, 1]


print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_rf_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, rf_probs)}')
print(f'Baseline ROC AUC: {roc_auc_score(y_test, [1 for _ in range(len(y_test))])}')


# In[33]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, grid_pred)).plot();


# In[35]:


from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
from subprocess import call
from IPython.display import Image
estimator = r_search.best_estimator_[1]
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


# ## Randomly dropping feature to see the performance of model ( bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1

# In[36]:


newdf = processed_data.drop(columns = 'education_level')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
rf3 = RandomForestClassifier(bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1
)
rf3.fit(X_train, y_train)
y_pred = rf3.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[37]:


newdf = processed_data.drop(columns = 'city_development_index')
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
rf3 = RandomForestClassifier(bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1
)
rf3.fit(X_train, y_train)
y_pred = rf3.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[38]:


newdf = processed_data.drop(['last_new_job','education_level'], axis=1)
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
rf3 = RandomForestClassifier(bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1
)
rf3.fit(X_train, y_train)
y_pred = rf3.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[39]:


newdf = processed_data.drop(['relevent_experience','education_level'], axis=1)
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
rf3 = RandomForestClassifier(bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1
)
rf3.fit(X_train, y_train)
y_pred = rf3.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[42]:


newdf = processed_data.drop(['education_level','experience'], axis=1)
X = newdf.drop(columns='target')
Y = newdf['target']
X_train,X_test,y_train,y_test = train_test_split(X, Y,test_size = 0.3, random_state = 1)
rf3 = RandomForestClassifier(bootstrap=False, max_depth=6, max_features=0.5, max_leaf_nodes=25, min_samples_split=10, n_estimators=50, random_state=1
)
rf3.fit(X_train, y_train)
y_pred = rf3.predict(X_test)
print('Model Accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred)).plot();


# In[43]:





# In[ ]:




