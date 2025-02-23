import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# โหลดข้อมูล
df = pd.read_csv("data/raw/house_prices.csv")

# จัดการข้อมูลที่หายไป
# เติมค่า NaN ในคอลัมน์ที่มีชนิดข้อมูลเป็นตัวเลขด้วยค่ามัธยฐาน
numeric_columns = df.select_dtypes(include=['number']).columns  # เลือกคอลัมน์ที่เป็นตัวเลข

# เติมค่า NaN ด้วยค่า median
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# แปลงข้อมูลให้เป็นตัวเลข
df = pd.get_dummies(df, drop_first=True)

# แยก Features & Target
X = df.drop(columns=["id","price"])
y = df["price"]

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# บันทึกข้อมูล
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Data processing complete!")
