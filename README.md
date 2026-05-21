# 💳 Real-Time Fraud Detection System

An AI-powered fraud detection system built using Machine Learning to identify fraudulent credit card transactions in real time.  
This project uses a Random Forest Classifier with SMOTE (Synthetic Minority Oversampling Technique) to handle highly imbalanced transaction data and improve fraud detection performance.

---

# 🚀 Project Overview

Credit card fraud detection is a major challenge in the financial industry because fraudulent transactions are extremely rare compared to normal transactions.

This project focuses on:

- Detecting fraudulent credit card transactions
- Handling imbalanced datasets using SMOTE
- Training a Machine Learning model using Random Forest
- Evaluating model performance using multiple metrics
- Generating fraud predictions and risk probabilities
- Saving trained models for future deployment

---

# 📂 Dataset

Dataset used:
- Credit Card Fraud Detection Dataset

Dataset Characteristics:
- Transactions made by European cardholders
- Highly imbalanced dataset
- Fraud cases are very rare
- Features are transformed using PCA (`V1` to `V28`)
- Includes:
  - `Time`
  - `Amount`
  - `Class`
    - `0` → Normal Transaction
    - `1` → Fraud Transaction

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

# ⚙️ Machine Learning Workflow

## 1️⃣ Data Preprocessing
- Loaded and cleaned dataset
- Checked missing values
- Scaled transaction amount and time features
- Prepared features and labels

---

## 2️⃣ Exploratory Data Analysis (EDA)
Performed:
- Fraud vs Normal transaction analysis
- Transaction amount analysis
- Transaction time distribution analysis
- Correlation heatmap visualization

---

## 3️⃣ Handling Imbalanced Data
Used:
- **SMOTE (Synthetic Minority Oversampling Technique)**

Purpose:
- Generate synthetic fraud samples
- Improve model learning on minority class
- Reduce model bias toward normal transactions

---

## 4️⃣ Model Training
Model Used:
- **Random Forest Classifier**

Key Parameters:
- `n_estimators = 100`
- `max_depth = 10`
- `class_weight = balanced`

---

## 5️⃣ Model Evaluation
Evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

# 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99.87% |
| Precision | 58.78% |
| Recall | 81.05% |
| F1-Score | 68.14% |
| ROC-AUC | 97.52% |

---

# 📈 Visualizations Included

- Transaction Amount Analysis
- Transaction Time Analysis
- Correlation Heatmap
- Feature Importance Graph
- Confusion Matrix
- ROC Curve

---

# 🧠 Feature Importance

The model identified the following features as highly important for fraud detection:

- V14
- V10
- V17
- V4
- V12
- V11

These features contributed most toward identifying fraudulent behavior.

---

# 🔍 Fraud Prediction System

The project includes:
- Single transaction prediction
- Batch transaction prediction
- Fraud probability estimation
- Risk level classification

Risk Levels:
- 🔴 High Risk
- 🟠 Medium Risk
- 🟡 Low Risk
- 🟢 Very Low Risk

---

# 💾 Model Saving

The trained model and scaler were saved using Joblib for future reuse.

Saved Files:
- `fraud_detection_model.pkl`
- `scaler.pkl`

---

# 📁 Project Structure

```bash
Real-Time-Fraud-Detection/
│
├── notebook/
│   └── fraud_detection.ipynb
│
├── outputs/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── screenshots/
│   ├── 02_amount_analysis.png
│   ├── 03_time_analysis.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_importance_corr.png
│   ├── 06_confusion_matrix_roc.png
│   └── 07_feature_importance.png
│
├── requirements.txt
└── README.md

```

# ▶️ How To Run The Project

## 1️⃣ Clone Repository

```bash
git clone https://github.com/r-sanjana/Real-Time-Fraud-Detection.git
```

---

## 2️⃣ Navigate To Project Folder

```bash
cd Real-Time-Fraud-Detection
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

Open:
```text
fraud_detection.ipynb
```

---

# 📌 Key Learnings

Through this project, I learned:

- Handling imbalanced datasets
- Data preprocessing techniques
- Fraud detection using Machine Learning
- Random Forest model training
- Model evaluation and visualization
- Risk probability analysis
- Saving and loading ML models

---

# 🎯 Future Improvements

Possible future enhancements:
- Real-time API integration
- Deep Learning models
- Live transaction monitoring
- Web dashboard deployment
- Advanced anomaly detection
- Real-time alert system

---

# 👩‍💻 Author

## Sanjana R

Git-Hub: https://github.com/r-sanjana
