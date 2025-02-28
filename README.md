# 🏡 House Price Prediction using Machine Learning

This project predicts house prices using machine learning techniques. It includes data preprocessing, model training, evaluation, and deployment. **DVC (Data Version Control)** is used for dataset versioning with **Google Drive** as remote storage.

## 🚀 Features
- Data preprocessing and cleaning
- Machine learning model training and evaluation
- House price prediction based on input features
- DVC for dataset tracking and versioning
- GitHub Actions for automation (optional)

## 📂 Project Structure
```
ML-HousePricePrediction/
│-- data/
│   ├── raw/                  
│   │   ├── house_prices.csv
│   ├── processed/            
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│-- models/                   
│   ├── house_price_model.pkl
│-- src/                      
│   ├── data_processing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference.py
│-- text_notebooks.ipynb 
│-- .dvc/                     
│-- dvc.yaml                  
│-- params.yaml               
│-- requirements.txt          
│-- README.md                 
```

## 📌 Getting Started
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SurakiatP/ML-HousePricePrediction.git
cd ML-HousePricePrediction
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Configure DVC with Google Drive
```bash
dvc remote add -d gdrive_remote gdrive://YOUR_GDRIVE_FOLDER_ID
git add .dvc/config
git commit -m "Set up DVC remote"
dvc push
```

### 4️⃣ Run the Project
- **Train the model**  
  ```bash
  python src/train_model.py
  ```
- **Evaluate the model**  
  ```bash
  python src/evaluate_model.py
  ```
- **Run inference**  
  ```bash
  python src/inference.py
  ```

## 📓 DVC Commands
- **Track data with DVC**  
  ```bash
  dvc add data/raw/house_prices.csv
  git add data/.gitignore data/raw/house_prices.csv.dvc
  git commit -m "Added data tracking with DVC"
  ```
- **Push data to remote storage**  
  ```bash
  dvc push
  ```
- **Pull data from remote storage**  
  ```bash
  dvc pull
  ```

## 🤝 Contributing
Feel free to fork the repository and submit a pull request with improvements.

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Contact
For questions, reach out at: [surakiat.0723@gmail.com] or connect on [LinkedIn](https://www.linkedin.com/in/surakiat-kansa-ard-171942351/)

---

