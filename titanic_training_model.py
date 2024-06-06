#!/usr/bin/env python
# coding: utf-8

# In[36]:


from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv(r"C:\Users\Matthew\Desktop\django-project\staticfiles\static_files\archive (1)\Titanic Dataset.csv")

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# A list with the all the different titles
titles = sorted(set([x for x in df.name.map(lambda x: get_title(x))]))


# Normalize the titles
def replace_titles(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ["Jonkheer","Don",'the Countess', 'Dona', 'Lady',"Sir"]:
        return 'Royalty'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title
    
# Lets create a new column for the titles
df['Title'] = df['name'].map(lambda x: get_title(x))
# train.Title.value_counts()
# train.Title.value_counts().plot(kind='bar')

# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'
df['Title'] = df.apply(replace_titles, axis=1)


df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['body'].fillna(df['body'].median(), inplace=True)
df['embarked'].fillna("S", inplace=True)
df.drop("cabin", axis=1, inplace=True)
df.drop("ticket", axis=1, inplace=True)
df.drop("name", axis=1, inplace=True)
df.drop('boat',axis=1,inplace=True)
df.drop("home.dest", axis=1, inplace=True)
df['sex'].replace(('male','female'), (0,1), inplace = True)
df['embarked'].replace(('S','C','Q'), (0,1,2), inplace = True)
df['Title'].replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'), (0,1,2,3,4,5,6,7), inplace = True)


predictors = df.drop(['survived'], axis=1)
target = df["survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)

filename = 'titanic_model.sav'
pickle.dump(randomforest, open(filename, 'wb'))

