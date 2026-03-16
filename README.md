# 🧥 Be Thrifty

**Be Thrifty** is an AI-powered fashion resale value predictor.  
The project estimates how much a thrifted or second-hand fashion item could sell for based on brand, category, condition, and other contextual signals.

The goal is to build a continuously improving pricing intelligence system for second-hand fashion.

---

# Overview

Second-hand fashion marketplaces like Depop, Grailed, and Mercari contain millions of listings with complex pricing dynamics.  
Be Thrifty uses machine learning to analyze patterns in these listings and predict resale prices.

The model learns from historical resale data and improves as additional datasets are added.

This project combines:

- data engineering
- feature engineering
- machine learning
- an interactive prediction dashboard

---

# Features

### AI Price Prediction
Predicts the resale value of fashion items using a trained machine learning model.

### Brand Intelligence
Recognizes brand tiers such as:

- luxury brands
- mid-tier brands
- fast fashion brands

### Feature Engineering
The model considers multiple factors:

- brand
- category
- condition
- size
- vintage status
- seasonal factors
- brand tier influence

### Interactive Dashboard
A Streamlit web application allows users to enter item details and instantly receive an estimated resale value.

### Real Marketplace Data
The system integrates resale marketplace datasets (e.g., Mercari) to learn real-world pricing behavior.

---

# Machine Learning Pipeline

The project follows a structured ML pipeline:

### 1. Data Preparation (`data_prep.py`)
- loads multiple datasets
- normalizes column names
- cleans missing values
- performs feature engineering
- encodes categorical features
- saves processed data

### 2. Model Training (`train.py`)
- splits data into training and test sets
- trains a Random Forest regression model
- evaluates performance using:
  - MAE
  - RMSE
  - R²
- saves the trained model

### 3. Prediction (`app.py`)
- loads the trained model
- collects user input
- generates real-time resale price predictions

---
# Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- Kaggle resale datasets
- Random Forest regression

---

# Project Structure

```
be-thrifty/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_prep.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   ├── raw_datasets/
│   └── processed/
│
└── models/
```

---

# Run the Project

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/thrift-price-predictor.git
cd thrift-price-predictor
```

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Prepare the dataset and train the model:

```
python src/data_prep.py
python src/train.py
```

Start the web application:

```
streamlit run app.py
```

Open the app in your browser:

```
http://localhost:8501
```

Enter item details such as brand, category, condition, and size to receive an estimated resale price.

---

# Model Accuracy & Data Growth

Be Thrifty is designed to improve as additional resale datasets are integrated.

The model currently trains on available resale listings and will become more accurate as more data is added.

Because resale markets are highly dynamic, increasing dataset size significantly improves prediction performance.

The pipeline is intentionally modular so new datasets can easily be added and the model retrained.

---

# Future Improvements

Planned upgrades include:

- larger resale datasets
- better brand recognition
- demand trend analysis
- price confidence intervals
- improved feature engineering
- model explainability with SHAP

The long-term vision is to build a full resale pricing intelligence system for second-hand fashion markets.

---

# License

This project is for educational and research purposes.



