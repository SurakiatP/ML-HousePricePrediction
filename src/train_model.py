import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# load hyperparameters
params = yaml.safe_load(open("params.yaml"))["train"]

# load train data from data/processed
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# build model
model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)

# fit model
model.fit(X_train, y_train.values.ravel())

# save model from models/house_price_model.pkl
joblib.dump(model, "models/house_price_model.pkl")

print("✅ Model training complete!")
