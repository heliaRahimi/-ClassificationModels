#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[2]:


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


# In[3]:


raw_data =  pd.read_csv("resources/aug_train.csv")
processed_data = preprocessing_data(raw_data)
training_df = processed_data.copy()
training_df.shape


# In[4]:


training_df.head()


# In[5]:


training_df.describe()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(training_df.drop('target',axis=1), 
                                                    training_df['target'], test_size=0.20, 
                                                    random_state=42)


# In[8]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , recall_score, precision_score


# In[9]:


#Usign SGD to model dataset
SGD_model = SGDClassifier(max_iter=1000, tol=1e-3) 


# In[10]:


SGD_model.fit(X_train, y_train)


# In[11]:


SGD_predictions = SGD_model.predict(X_test)


# In[12]:


print("confusion matrix is: \n", confusion_matrix(y_test,SGD_predictions))


# In[13]:


print("Accuracy score is: " ,accuracy_score(SGD_predictions, y_test), '\nPrecision is : ',precision_score(y_test,SGD_predictions),'\nRecall score is: ' ,recall_score(y_test,SGD_predictions))


# In[14]:


#Usign GridSearchCV to tune parameters
from sklearn.model_selection import GridSearchCV


# In[15]:


param_grid = {'loss': ['hinge', 'log', 'perceptron','squared_error','huber','epsilon_insensitive'], 'penalty': ['l1', 'l2','elasticnet'], 'penalty': ['l1', 'l2'], 'alpha': [0.0001,0.1,100.0],'random_state': [1, None], 'max_iter': [100,1000,5000]}


# In[16]:


grid = GridSearchCV(SGDClassifier(),param_grid,refit=True,verbose=3)


# In[17]:


grid.fit(X_train, y_train)


# In[18]:


print("the best combination is :" ,grid.best_estimator_ )


# In[19]:


grid_predictions = grid.predict(X_test)


# In[20]:


print("confusion matrix is: \n", confusion_matrix(y_test,SGD_predictions))


# In[21]:


print("Accuracy score is: " ,accuracy_score(SGD_predictions, y_test), '\nPrecision is : ',precision_score(y_test,SGD_predictions),'\nRecall score is: ' ,recall_score(y_test,SGD_predictions))


# Grid search parameter tuning did not improve the initial model.

# In[22]:


# Drop education_level columns
training_df = processed_data.drop('education_level',axis=1)
training_df.head()


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(training_df.drop('target',axis=1), 
                                                    training_df['target'], test_size=0.20, 
                                                    random_state=42)


# In[24]:


SGD_model = SGDClassifier(max_iter=100) #Using tuned parameteres
SGD_model.fit(X_train, y_train)


# In[25]:


SGD_predictions = SGD_model.predict(X_test)
print("confusion matrix is: \n", confusion_matrix(y_test,SGD_predictions))


# In[26]:


print("Accuracy score is: " ,accuracy_score(SGD_predictions, y_test))


# In[27]:



print("Accuracy score is: " ,accuracy_score(SGD_predictions, y_test), '\nPrecision is : ',precision_score(y_test,SGD_predictions),'\nRecall score is: ' ,recall_score(y_test,SGD_predictions))


# Accuracy, precision and recall improved.
