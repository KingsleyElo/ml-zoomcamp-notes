import pickle
from transformers import df_to_dict
import pandas as pd

input_file = 'model_C=1.bin'

with open (input_file, 'rb') as f_in:
    model = pickle.load(f_in)

customer =  pd.DataFrame([{'tenure': 41,
  'monthlycharges': 79.85,
  'totalcharges': 3320.75,
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'no',
  'deviceprotection': 'yes',
  'techsupport': 'yes',
  'streamingtv': 'yes',
  'streamingmovies': 'yes',
  'contract': 'one_year',
  'paperlessbilling': 'yes',
  'paymentmethod': 'bank_transfer_(automatic)'}])

y_pred = model.predict_proba(customer)[0, 1]

print('input', customer.to_dict(orient='records'))
print('churn Probability', y_pred)