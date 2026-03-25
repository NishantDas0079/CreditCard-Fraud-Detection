import os
os.chdir(r"C:\Users\Nishant\Downloads\NDxGenius\fraud_detection")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/creditcard.csv')

# Shape and info
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes.value_counts())
print("\nMissing values:", df.isnull().sum().sum())

# Class distribution
print("\nClass distribution:")
print(df['Class'].value_counts())

# Plot class distribution
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# Transaction amount statistics
print("\nAmount statistics:")
print(df['Amount'].describe())

# Plot amount distribution (log scale)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df[df['Class']==0]['Amount'], bins=50, log_scale=True)
plt.title('Non-Fraud Amounts')
plt.subplot(1,2,2)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, log_scale=True)
plt.title('Fraud Amounts')
plt.tight_layout()
plt.show()