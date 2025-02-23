import joblib
import pandas as pd
import sys

# โหลดโมเดลโดยใช้ joblib
model_path = "models/house_price_model.pkl"
model = joblib.load(model_path)

# ตรวจสอบว่าโหลดโมเดลถูกต้อง
print(f"Loaded model type: {type(model)}")  # ควรแสดง <class 'sklearn.ensemble.RandomForestRegressor'>

# รับค่า Input จาก Command Line
square_feet = int(sys.argv[1])
bedrooms = int(sys.argv[2])
bathrooms = int(sys.argv[3])
location = sys.argv[4]
year_built = int(sys.argv[5])

# One-Hot Encoding สำหรับ location
locations = ["Countryside", "Downtown", "Mountain", "Suburban"]
location_encoded = {f"location_{loc}": 0 for loc in locations}
if f"location_{location}" in location_encoded:
    location_encoded[f"location_{location}"] = 1

# สร้าง DataFrame สำหรับอินพุต
input_data = {
    "square_feet": [square_feet],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "year_built": [year_built],
    **location_encoded  
}

input_df = pd.DataFrame(input_data)

# ทำนายราคา
predicted_price = model.predict(input_df)[0]
print(f"Predicted House Price: ${predicted_price:,.2f}")
