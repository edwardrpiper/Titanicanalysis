
# coding: utf-8

# In[187]:


# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

data_train = pd.read_csv('train.csv')
y = data_train.pop('Survived')
data_test = pd.read_csv('test.csv')




avgAge = data_train.Age.mean()
data_train.Age = data_train.Age.fillna(value=avgAge)
data_test.Age = data_test.Age.fillna(value=avgAge)



data_train.drop(["Name","Ticket","PassengerId"], axis=1, inplace=True)
data_test.drop(["Name","Ticket","PassengerId"], axis=1, inplace=True)



char_cabin = data_train['Cabin'].astype(str)
new_cabin = np.array([cabin[0] for cabin in char_cabin])
new_cabin = pd.Categorical(new_cabin)
data_train['Cabin'] = new_cabin

char_cabin2 = data_test['Cabin'].astype(str)
new_cabin2 = np.array([cabin[0] for cabin in char_cabin2])
new_cabin2 = pd.Categorical(new_cabin2)
data_test['Cabin'] = new_cabin

data_train = pd.get_dummies(data_train, prefix='is_')

data_test2 = pd.get_dummies(data_test, prefix='is_')





model2 = RandomForestRegressor(n_estimators=1000,oob_score=True, random_state=42)
model2.fit(data_train,y)


# In[ ]:


model2.oob_score_


