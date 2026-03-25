# 🏦 Credit Card Fraud Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.ai/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7)](https://render.com/)
[![GitHub last commit](https://img.shields.io/github/last-commit/NishantDas0079/CreditCard-Fraud-Detection)](https://github.com/NishantDas0079/CreditCard-Fraud-Detection)

**A machine learning project to detect fraudulent credit card transactions with a banking‑themed interactive dashboard.**

This project combines state‑of‑the‑art classification models (XGBoost, Random Forest, Logistic Regression) with a beautiful Streamlit frontend. Users can enter transaction details and instantly receive a fraud probability, helping financial institutions or individuals flag suspicious activity in real time.

![App Screenshot](screenshot.png) <!-- Add a screenshot of your dashboard -->

---

## ✨ Features

- **🎯 Real‑time prediction** – Enter transaction amount and time, get fraud probability instantly.
- **📊 Interactive dashboard** – Banking‑themed UI with risk meter, verdict cards, and detailed breakdown.
- **🤖 Multiple models** – XGBoost, Random Forest, and Logistic Regression (best model saved).
- **⚖️ Handles class imbalance** – SMOTE oversampling used to balance the training set.
- **📁 Model persistence** – Trained models and scaler are saved and reloaded seamlessly.
- **📈 Evaluation metrics** – ROC curves, confusion matrices, and classification reports available.
- **🔧 Easy to extend** – Retrain with custom hyperparameters or new data.

---

## 🛠️ Tech Stack

| Component        | Technology                         |
|------------------|------------------------------------|
| Frontend         | [Streamlit](https://streamlit.io/) |
| ML Models        | XGBoost, Random Forest, Logistic Regression |
| Data Handling    | Pandas, NumPy                      |
| Preprocessing    | Scikit‑learn, SMOTE (imbalanced‑learn) |
| Visualisation    | Matplotlib, Seaborn, Plotly        |
| Deployment       | Render (or Streamlit Cloud)        |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git
- A Kaggle account (to download the dataset)

# STEPS
# 1. Clone the Repository
```
git clone https://github.com/NishantDas0079/CreditCard-Fraud-Detection.git
cd CreditCard-Fraud-Detection
```

# 2. Create and activate virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

# 3. Install dependencies
```
pip install -r requirements.txt
```

# 4. Download the dataset

The dataset (creditcard.csv) is not included due to its size (~130 MB).

Download it from Kaggle and place it in the data/ folder.

If you don't have a Kaggle account, you can also obtain it via the UCI Machine Learning Repository.

# 5. Run the App
```
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

# 🎯 Usage
Enter the Transaction Amount and Time (seconds since the first transaction in the dataset).

Click "Analyze Transaction".

View the fraud probability, risk meter, and verdict (Fraud / Legitimate).

Expand the "Detailed analysis" section to see the raw model output and heuristic risk factor.

# 🧠 Model Training
The training pipeline consists of:

Data exploration – Checking class imbalance, missing values, and distributions.

Preprocessing – Scaling `Amount` and `Time` with `StandardScaler`.

Train/test split – 80/20 split, stratified to preserve class ratio.

SMOTE – Synthetic Minority Over‑sampling applied to the training set to balance fraud/non‑fraud classes.

Model training – Logistic Regression, Random Forest, and XGBoost are trained and evaluated.

Model selection – XGBoost performed best (AUC ≈ 0.98) and is saved for inference.

Evaluation – ROC curves, confusion matrices, and classification reports are generated.

# 📊 Dataset
The dataset used is the Credit Card Fraud Detection dataset from Kaggle. It contains 284,807 transactions, of which only 492 are fraudulent (0.172%). Features are:

`Time` – seconds elapsed between the first transaction and the current one.

`V1` … `V28` – PCA‑transformed components (confidential).

`Amount` – transaction amount.

`Class` – target: 0 = legitimate, 1 = fraud.

# 📄 License
This project is licensed under the `Apache License 2.0`. See the `LICENSE` file for details.
