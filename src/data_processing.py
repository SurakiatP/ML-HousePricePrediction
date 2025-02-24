import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv("data/raw/house_prices.csv")

numeric_columns = df.select_dtypes(include=['number']).columns  
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=["id","price"])
y = df["price"]

# split data for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Data processing complete!")
