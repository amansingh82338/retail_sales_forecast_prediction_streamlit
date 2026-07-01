# 📈 Predictive Sales Forecasting Using Machine Learning

An end-to-end sales forecasting application that predicts future **weekly retail sales** by analyzing historical sales data, store information, and external economic factors such as holidays, temperature, fuel prices, CPI, and unemployment. Built with **Python**, **Machine Learning**, and **Streamlit**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dashboard Preview](#-dashboard-preview)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

This project delivers a complete machine learning pipeline for retail sales forecasting:

1. **Data Preprocessing** — cleaning raw datasets by handling missing values, removing duplicates, correcting data types, and merging multiple sources into a single consolidated dataset.
2. **Exploratory Data Analysis (EDA)** — understanding sales trends, seasonal patterns, store/department performance, and the impact of holidays and economic indicators.
3. **Feature Engineering** — creating meaningful variables such as year, month, quarter, week, day of week, lag sales, rolling averages, and encoded categorical features.
4. **Model Training & Comparison** — training and evaluating four regression models to identify the best performer.
5. **Interactive Dashboard** — a Streamlit app for exploring data, visualizing trends, comparing models, and generating live sales predictions.

---

## ✨ Features

- 🧹 Automated data cleaning and merging pipeline
- 📊 Rich exploratory data analysis with trend, seasonality, and holiday-impact visualizations
- 🛠️ Engineered time-based and lag/rolling-average features
- 🤖 Four trained ML models with side-by-side performance comparison
- 🏆 Automatic selection and export of the best-performing model
- 📉 Evaluation via MAE, RMSE, and R² Score
- 🖥️ Interactive Streamlit dashboard for exploration and live forecasting

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Persistence | Joblib |
| Dashboard | Streamlit |

---

## 🔄 Project Workflow

```
Raw Data
   ↓
Data Cleaning
   ↓
Data Merging
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Model Training (4 ML Models)
   ↓
Model Evaluation
   ↓
Sales Forecast Prediction
   ↓
Interactive Streamlit Dashboard
```

---

## 📁 Project Structure

```
predictive_sales_forecast/
│
├── data/
│   ├── raw/                       # Original, unprocessed datasets
│   ├── processed/                 # Cleaned individual datasets
│   └── merged/
│       └── merged_sales.csv       # Final consolidated dataset
│
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl             # Best-performing model (auto-selected)
│   └── scaler.pkl                 # Fitted StandardScaler
│
├── reports/
│   ├── model_comparison.csv
│   ├── metrics.csv
│   ├── feature_importance.csv
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── notebooks/                     # EDA & feature engineering notebooks
│
├── app.py                         # Streamlit dashboard
├── train_model.py                 # Model training & evaluation script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/predictive_sales_forecast.git
   cd predictive_sales_forecast
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Train the models

Run the training script to preprocess data, train all four models, evaluate them, and export models/reports:

```bash
python train_model.py
```

This generates the `models/` and `reports/` directories shown in the project structure above.

### 2. Launch the dashboard

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) to explore the dataset, view sales trends, compare model performance, and generate predictions.

---

## 📊 Model Performance

Four models were trained and evaluated on the merged dataset using **MAE**, **RMSE**, and **R² Score**:

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | — | — | — |
| Decision Tree Regressor | — | — | — |
| Random Forest Regressor | — | — | — |
| XGBoost Regressor | — | — | — |

> Full results are available in [`reports/model_comparison.csv`](reports/model_comparison.csv) after running `train_model.py`. The best-performing model is automatically saved as `models/best_model.pkl`.

---

## 🖥️ Dashboard Preview

The Streamlit dashboard allows users to:

- Explore raw and processed sales data
- Visualize sales trends across stores, departments, and time
- Analyze the impact of holidays, temperature, fuel prices, CPI, and unemployment
- Compare performance across all trained models
- Generate future sales predictions interactively

*(Add screenshots or a GIF of the dashboard here once available.)*

---

## 🔮 Future Improvements

- Incorporate deep learning models (e.g., LSTM) for sequential forecasting
- Add hyperparameter tuning (GridSearchCV / Optuna) for further model optimization
- Deploy the dashboard to a cloud platform (Streamlit Community Cloud, AWS, or Azure)
- Add automated model retraining pipeline
- Include confidence intervals for predictions

---

## 👤 Author

**[Your Name]**
Data Analyst / Machine Learning Enthusiast

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Name](https://linkedin.com/in/your-profile)

---

*This project demonstrates a complete machine learning workflow — from raw data preprocessing to deploying an interactive forecasting application — and serves as a practical portfolio project for showcasing data analysis, machine learning, and dashboard development skills.*
