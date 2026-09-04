# 📈 Retail Demand Forecasting

An end-to-end time-series forecasting project to predict daily unit demand across multiple stores and product categories, using calendar signals and historical sales patterns.
The project includes a full feature-engineering + model-comparison pipeline and a Streamlit app for interactive predictions.

## 🚀 Features
- Synthetic multi-store, multi-category daily sales dataset with trend, seasonality, and promotional spikes
- Time-series feature engineering: lag features (1/7/14/28-day) and rolling mean/std windows
- Time-based train/test split (no shuffling — respects temporal order, avoids data leakage)
- Model comparison: Naive baseline, Linear Regression, Random Forest, XGBoost
- Evaluation with MAE, RMSE, and MAPE
- Feature importance and actual-vs-predicted visualizations
- Interactive Streamlit app for real-time demand prediction

## 🧠 Forecasting Pipeline
1. Data cleaning and calendar feature extraction (day of week, month, weekend flag, quarter)
2. Lag feature generation per store/category series (demand N days ago)
3. Rolling statistics (7/14/28-day mean & std), shifted to avoid leaking same-day values
4. One-hot encoding for store/category via `ColumnTransformer`
5. Time-based split: last 60 days held out as test set
6. Model training and comparison across four approaches
7. Best model saved for inference; feature importances and prediction plots exported

## 📊 Results (Test Set — Last 60 Days)
| Model               | MAE  | RMSE  | MAPE  |
|---------------------|------|-------|-------|
| Naive (lag-7)        | 12.14 | 17.20 | 7.00% |
| Random Forest         | 7.41  | 9.33  | 4.31% |
| XGBoost               | 7.34  | 9.53  | 4.24% |
| Linear Regression     | 7.21  | 9.24  | 4.27% |

All three ML models cut forecasting error roughly in half versus the naive "same day last week" baseline.

## 🖥️ Streamlit App
Users can input:
- Store and product category
- Calendar details (date, day of week, promotion flag)
- Recent sales history (lag and rolling statistics)

The app outputs:
- Predicted units sold for that day

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib
- Streamlit

## ▶️ How to Run
```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python generate_data.py

# 2. Train models and generate evaluation reports
python train.py

# 3. Launch the interactive app
streamlit run app.py
```

## 📁 Project Structure
```
├── generate_data.py      # Synthetic data generation
├── train.py               # Feature engineering, model training & evaluation
├── app.py                  # Streamlit inference app
├── data/
│   └── sales_data.csv
├── models/
│   ├── xgb_demand_model.pkl
│   └── rf_demand_model.pkl
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
└── requirements.txt
```
