import os
os.chdir(r"C:\Users\Nishant\Downloads\NDxGenius\fraud_detection")

import joblib
import pandas as pd
import numpy as np

def load_model(model_path='models/fraud_detector.pkl'):
    return joblib.load(model_path)

def load_scaler(scaler_path='models/scaler.pkl'):
    return joblib.load(scaler_path)

def predict_transaction(model, scaler, transaction_dict):
    """
    transaction_dict: dict with keys matching feature names (including 'Time' and 'Amount')
    Returns: fraud probability and class prediction.
    """
    # Convert to DataFrame
    X_new = pd.DataFrame([transaction_dict])
    # Scale 'Amount' and 'Time' using the fitted scaler
    X_new[['Amount', 'Time']] = scaler.transform(X_new[['Amount', 'Time']])
    prob = model.predict_proba(X_new)[0, 1]
    pred = 1 if prob > 0.5 else 0
    return pred, prob

if __name__ == '__main__':
    # Example: a sample transaction (replace with real values)
    # You need to provide values for all 30 features (V1..V28, Time, Amount)
    # For demo, we'll create a dummy transaction with zeros for PCA features
    sample_transaction = {
        'Time': 0,
        'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'V5': 0, 'V6': 0, 'V7': 0, 'V8': 0,
        'V9': 0, 'V10': 0, 'V11': 0, 'V12': 0, 'V13': 0, 'V14': 0, 'V15': 0,
        'V16': 0, 'V17': 0, 'V18': 0, 'V19': 0, 'V20': 0, 'V21': 0, 'V22': 0,
        'V23': 0, 'V24': 0, 'V25': 0, 'V26': 0, 'V27': 0, 'V28': 0,
        'Amount': 100.0
    }
    
    model = load_model()
    scaler = load_scaler()
    pred, prob = predict_transaction(model, scaler, sample_transaction)
    print(f"Prediction: {'Fraud' if pred else 'Legitimate'}")
    print(f"Fraud Probability: {prob:.2%}")