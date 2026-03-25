import os
os.chdir(r"C:\Users\Nishant\Downloads\NDxGenius\fraud_detection")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess(test_size=0.2, use_smote=True, random_state=42):
    # Load data
    df = pd.read_csv('data/creditcard.csv')
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    
    # Split data (stratify to keep class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Apply SMOTE only to training set
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: Train shape {X_train_res.shape}")
        print(f"Class distribution:\n{y_train_res.value_counts()}")
        return X_train_res, X_test, y_train_res, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test, scaler

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")