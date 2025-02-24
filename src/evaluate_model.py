import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load test data from data/processed
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# load model
model = joblib.load("models/house_price_model.pkl")

# predict
y_pred = model.predict(X_test)

# MSE/MAE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"📊 Model Evaluation:\n - MAE: {mae}\n - MSE: {mse}")
