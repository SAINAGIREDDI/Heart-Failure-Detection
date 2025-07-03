# 🫀 Heart Failure Detection using SMOTE and Machine Learning

A robust full-stack data science project focused on the early detection of **chronic heart failure** using **SMOTE (Synthetic Minority Over-sampling Technique)** and a range of machine learning models. This project enhances prediction accuracy through **balanced datasets** and **algorithmic evaluation**.

---

## 🚀 Features

### 📊 Data Science & Machine Learning

* SMOTE for handling class imbalance
* Feature extraction and normalization
* Models used: Logistic Regression, Decision Trees, Random Forest, SVM, AdaBoost, XGBoost, KNN
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC Curve, Confusion Matrix

### 🧠 Research Contributions

* Novel **Principal Component Heart Failure (PCHF)** feature engineering
* Integration of **exercise therapy insights** and **ECG signal analysis**
* Cross-validation methodology for model robustness

---

## 🛠️ Tech Stack

| Layer         | Technology                              |
| ------------- | --------------------------------------- |
| Programming   | Python, Jupyter Notebook                |
| ML Libraries  | scikit-learn, XGBoost, SMOTE (imblearn) |
| Data Handling | Pandas, NumPy                           |
| Visualization | Matplotlib, Seaborn                     |
| Other Tools   | SMOTE, ROC-AUC, Confusion Matrix        |

---

## 📂 Project Structure

```
HeartFailure-Detection-ML/
├── data/                      # Dataset files (.csv, .xlsx, etc.)
├── notebooks/                 # Jupyter notebooks with exploratory analysis and models
│   ├── preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── models/                    # Saved ML models (optional .pkl/.joblib)
├── results/                   # Confusion matrices, ROC curves, performance metrics
├── figures/                   # Plots and graphs for paper
├── manuscript/                # DOCX or PDF of the published paper
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # To exclude unnecessary files
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

* Python 3.8+
* pip or conda

### 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/HeartFailure-Detection-ML.git
   cd HeartFailure-Detection-ML
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # or env\Scripts\activate on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

---

## 📈 Results

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| XGBoost       | 0.974    | 0.926     | 0.909  | 0.917    |
| AdaBoost      | 0.859    | 0.780     | 0.830  | 0.804    |
| Decision Tree | 0.852    | 0.755     | 0.851  | 0.800    |
| SVM           | 0.852    | 0.865     | 0.681  | 0.762    |
| Random Forest | 0.852    | 0.829     | 0.723  | 0.773    |
| KNN           | 0.830    | 0.833     | 0.638  | 0.723    |

> 📊 **XGBoost** achieved the highest performance across all metrics.

---

## 📌 Future Enhancements

* ✅ Live ECG data integration
* ✅ GUI-based prediction interface
* 💬 Patient-doctor chat recommendation system
* 🔐 Federated learning for privacy-preserving predictions
* 📦 Integration with hospital EHR systems

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with proper citation.

