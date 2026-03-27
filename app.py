# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")
st.title("📊 Sales Prediction Dashboard")
st.markdown("This dashboard predicts sales using Linear Regression, Random Forest, and XGBoost models.")

# ----------------------------
# Load your CSV automatically
# ----------------------------
try:
    df = pd.read_csv("suheba_dataset.csv", encoding="latin1")
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    st.error("❌ suheba_dataset.csv not found in the folder. Please make sure it is in the same folder as app.py.")
    st.stop()

st.write("Preview of your data:", df.head())

# Drop unnecessary columns
drop_cols = [
    "Row ID","Order ID","Customer ID","Customer Name",
    "Product ID","Product Name","Order Date","Ship Date","Postal Code"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Check if 'Sales' exists
if "Sales" not in df.columns:
    st.error("Your dataset must contain a 'Sales' column.")
    st.stop()

# Split features and target
X = df.drop(columns=["Sales"])
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Model Training & Predictions")

# Train models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1, objective='reg:squarederror'
).fit(X_train, y_train)

# Predict
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

st.success("✅ Models trained successfully!")

# Metrics function
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Show metrics table
metrics_df = pd.DataFrame({
    "Model": ["Linear Regression","Random Forest","XGBoost"],
    "MAE": [metrics(y_test, y_pred_lr)[0],
            metrics(y_test, y_pred_rf)[0],
            metrics(y_test, y_pred_xgb)[0]],
    "RMSE": [metrics(y_test, y_pred_lr)[1],
             metrics(y_test, y_pred_rf)[1],
             metrics(y_test, y_pred_xgb)[1]],
    "R2 Score": [metrics(y_test, y_pred_lr)[2],
                 metrics(y_test, y_pred_rf)[2],
                 metrics(y_test, y_pred_xgb)[2]]
})
st.subheader("📋 Model Comparison Table")
st.dataframe(metrics_df)

# Plot Actual vs Predicted
st.subheader("📈 Actual vs Predicted Sales")
fig, axes = plt.subplots(1, 3, figsize=(18,5))
min_val, max_val = y_test.min(), y_test.max()

# Linear Regression plot
axes[0].scatter(y_test, y_test, color='black', label='Actual Sales', alpha=0.6)
axes[0].scatter(y_test, y_pred_lr, color='pink', label='Predicted Sales', alpha=0.6)
axes[0].plot([min_val,max_val],[min_val,max_val],'r--')
axes[0].set_title("Linear Regression")
axes[0].set_xlabel("Actual Sales")
axes[0].set_ylabel("Sales")
axes[0].legend()

# Random Forest plot
axes[1].scatter(y_test, y_test, color='black', label='Actual Sales', alpha=0.6)
axes[1].scatter(y_test, y_pred_rf, color='lightblue', label='Predicted Sales', alpha=0.6)
axes[1].plot([min_val,max_val],[min_val,max_val],'r--')
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Actual Sales")
axes[1].set_ylabel("Sales")
axes[1].legend()

# XGBoost plot
axes[2].scatter(y_test, y_test, color='black', label='Actual Sales', alpha=0.6)
axes[2].scatter(y_test, y_pred_xgb, color='yellow', label='Predicted Sales', alpha=0.6)
axes[2].plot([min_val,max_val],[min_val,max_val],'r--')
axes[2].set_title("XGBoost")
axes[2].set_xlabel("Actual Sales")
axes[2].set_ylabel("Sales")
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

st.info("💡 The app automatically uses 'suheba_dataset.csv'. No uploading needed!")