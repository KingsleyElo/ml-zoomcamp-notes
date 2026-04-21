
import requests

host = 'churn-serving-env.eba-mz2a3q36.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

# %%
customer = {
  "tenure": 12,
  "monthlycharges": 79.85,
  "totalcharges": (1* 79.85),
  "gender": 'female',
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "no",
  "phoneservice": "yes",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "yes",
  "onlinebackup": "no",
  "deviceprotection": "yes",
  "techsupport": "yes",
  "streamingtv": "yes",
  "streamingmovies": "yes",
  "contract": "one_year",
  "paperlessbilling": "yes",
  "paymentmethod": "bank_transfer_(automatic)"
}

# %%
response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print(f'sending promo email to customer')
else:
    print('not sending promo email to customer')

# %% [markdown]
# to deploy use the below
# 
# ```bash
# waitress-serve --listen=0.0.0.0:9696 predict:app
# ```


