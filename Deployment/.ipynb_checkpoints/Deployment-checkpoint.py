#!/usr/bin/env python
# coding: utf-8

# import the necessary libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[2]:

print("data loaded")
df = pd.read_csv('C:/Users/osaze/Desktop/Git Projects/ml-zoomcamp-notes/Logistics regression/Telco-Customer-Churn.csv')
df.head()


print("performing data preprocessing")
# Data Preprocessing

# convert column headers to lower case and replace space with _
# convert categorical column values to lower case and replace space with _

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# convert total charges to number and replace nulls with 0
df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')
df.totalcharges = df.totalcharges.fillna(0)

# convert yes to 1 and no to 0 and convert the datatype to int
df.churn = (df.churn == 'yes').astype(int)



print("performing two way split with cross validation")
# split the dataset into train, validation and test sets
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

df_full_train = df_full_train.reset_index(drop=True)



numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
full_list = numerical + categorical


# Training the model

print("Training the model")

def df_to_dict(X):
    return X.to_dict(orient="records") 

model = make_pipeline(FunctionTransformer(df_to_dict, validate=False), 
                      DictVectorizer(sparse = False), 
                      LogisticRegression(max_iter =5000))


# implement kfold cross validation, to ensure that the model is trained on the best parameter
print("implementing cross validation with 5 splits")
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# Define parameter grid
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV (
    model,
    param_grid,
    cv=kfold,
    scoring = 'roc_auc',
    refit=True
)

grid.fit(df_full_train[full_list], df_full_train.churn.values)

print("Cross Validation Results...")
print("Best C:", grid.best_params_)
print("Best score:", grid.best_score_)


# Predict Based on the best model parameters
best_model = grid.best_estimator_
y_pred = best_model.predict_proba(df_test[full_list])[:,1]

y_test = df_test['churn'].values
auc = roc_auc_score(y_test, y_pred)

print(f"auc={auc}")


# **save the model**

output_file = f"model_C={grid.best_params_['logisticregression__C']}.bin"
output_file


with open(output_file, 'wb') as f_out:
    pickle.dump(best_model, f_out)

print(f'this model is saved to {output_file}')