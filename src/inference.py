import joblib
import pandas as pd
import sys

# load model by joblib
model_path = "models/house_price_model.pkl"
model = joblib.load(model_path)

# check model
print(f"Loaded model type: {type(model)}")  # <class 'sklearn.ensemble.RandomForestRegressor'>

# get Input from Command Line
square_feet = int(sys.argv[1])
bedrooms = int(sys.argv[2])
bathrooms = int(sys.argv[3])
location = sys.argv[4]
year_built = int(sys.argv[5])

# One-Hot Encoding for location
locations = ["Countryside", "Downtown", "Mountain", "Suburban"]
location_encoded = {f"location_{loc}": 0 for loc in locations}
if f"location_{location}" in location_encoded:
    location_encoded[f"location_{location}"] = 1

# build DataFrame for input
input_data = {
    "square_feet": [square_feet],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "year_built": [year_built],
    **location_encoded  
}

input_df = pd.DataFrame(input_data)

# prediction
predicted_price = model.predict(input_df)[0]
print(f"Predicted House Price: ${predicted_price:,.2f}")
