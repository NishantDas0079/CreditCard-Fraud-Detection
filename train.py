import os
os.chdir(r"C:\Users\Nishant\Downloads\NDxGenius\fraud_detection")

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_and_preprocess

def train_models():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'model': model,
            'report': report,
            'auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"{name} - AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
    
    return results, X_test, y_test

if __name__ == '__main__':
    results, X_test, y_test = train_models()
    # Save the best model (XGBoost often performs best)
    best_model = results['XGBoost']['model']
    joblib.dump(best_model, 'models/fraud_detector.pkl')
    print("Best model saved as 'models/fraud_detector.pkl'")