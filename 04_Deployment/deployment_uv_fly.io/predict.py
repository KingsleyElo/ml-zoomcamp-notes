import pickle

## loading a model with pickle
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


customer = {
    'gender': 'Male',
    'seniorcitizen': 0,
    'partner': 'No',
    'dependents': 'Yes',
    'phoneservice': 'No',
    'multiplelines': 'No phone service',
    'internetservice': 'DSL',
    'onlinesecurity': 'No',
    'onlinebackup': 'Yes',
    'deviceprotection': 'No',
    'techsupport': 'No',
    'streamingtv': 'No',
    'streamingmovies': 'No',
    'contract': 'Month-to-month',
    'paperlessbilling': 'Yes',
    'paymentmethod': 'Electronic check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 129.85
}

churn = pipeline.predict_proba(customer)[0,1]


print(f'probability of churning = {churn}')
if churn >= 0.5:
    print('send email with promo')
else:
    print("don't do anything")

