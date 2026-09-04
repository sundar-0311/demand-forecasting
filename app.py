"""
Streamlit app for interactive demand forecasting.
Loads the trained XGBoost pipeline and lets the user simulate a day's
conditions (store, category, calendar info, recent sales history) to
get a predicted units-sold forecast.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="centered")

st.title("📈 Retail Demand Forecasting")
st.write(
    "Predicts expected **units sold** for a given store/category on a given day, "
    "based on calendar signals and recent sales history (lag & rolling features)."
)

model = joblib.load("models/xgb_demand_model.pkl")

st.header("Store & Date Details")
col1, col2 = st.columns(2)
with col1:
    store = st.selectbox("Store", ["Store_A", "Store_B", "Store_C"])
    category = st.selectbox("Category", ["Electronics", "Groceries", "Apparel"])
    month = st.slider("Month", 1, 12, 6)
    day_of_week = st.selectbox(
        "Day of week", options=list(range(7)),
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
    )
with col2:
    day_of_month = st.slider("Day of month", 1, 31, 15)
    week_of_year = st.slider("Week of year", 1, 52, 24)
    quarter = (month - 1) // 3 + 1
    promo = st.checkbox("Promotion running today?")

is_weekend = int(day_of_week >= 5)

st.header("Recent Sales History (for lag/rolling features)")
st.caption("In production these would come from your database automatically — entered manually here for demo purposes.")

col3, col4 = st.columns(2)
with col3:
    lag_1 = st.number_input("Units sold 1 day ago", min_value=0, value=120)
    lag_7 = st.number_input("Units sold 7 days ago", min_value=0, value=115)
    lag_14 = st.number_input("Units sold 14 days ago", min_value=0, value=118)
    lag_28 = st.number_input("Units sold 28 days ago", min_value=0, value=110)
with col4:
    rolling_mean_7 = st.number_input("7-day rolling avg", min_value=0.0, value=118.0)
    rolling_mean_14 = st.number_input("14-day rolling avg", min_value=0.0, value=116.0)
    rolling_mean_28 = st.number_input("28-day rolling avg", min_value=0.0, value=114.0)

rolling_std_7 = st.number_input("7-day rolling std dev", min_value=0.0, value=8.0)
rolling_std_14 = st.number_input("14-day rolling std dev", min_value=0.0, value=9.0)
rolling_std_28 = st.number_input("28-day rolling std dev", min_value=0.0, value=10.0)

if st.button("Predict Demand", type="primary"):
    input_df = pd.DataFrame([{
        "store": store,
        "category": category,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "promo": int(promo),
        "day_of_month": day_of_month,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_7": rolling_std_7,
        "rolling_std_14": rolling_std_14,
        "rolling_std_28": rolling_std_28,
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"### Predicted Demand: **{prediction:.0f} units**")

    if promo:
        st.info("Promotion flag is boosting the forecast — compare with promo unchecked to see the lift.")
