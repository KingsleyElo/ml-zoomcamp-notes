import pickle
from transformers import df_to_dict
import pandas as pd
from flask import Flask, request, jsonify

input_file = 'model_C=1.bin'

with open (input_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('Customer Churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    customer = [customer]
    customer = pd.DataFrame(customer)
    
    y_pred = model.predict_proba(customer)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)