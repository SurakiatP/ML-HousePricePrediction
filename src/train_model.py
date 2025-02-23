import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# โหลด hyperparameters
params = yaml.safe_load(open("params.yaml"))["train"]

# โหลดข้อมูลที่ผ่านการประมวลผลแล้ว
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# สร้างโมเดล
model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)

# เทรนโมเดล
model.fit(X_train, y_train.values.ravel())

# บันทึกโมเดล
joblib.dump(model, "models/house_price_model.pkl")

print("✅ Model training complete!")
