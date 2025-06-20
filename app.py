import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io
from datetime import datetime

# Set Streamlit dark theme with custom styling
st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")

# Inject dark style override
st.markdown("""
    <style>
        body, .stApp { background-color: #0e1117; color: white; }
        .css-1d391kg { color: white; }
        .stDataFrame th, .stDataFrame td { background-color: #1e222a; color: white; }
        .block-container { padding: 1.5rem 2rem; }
        .metric { background-color: #1f2937 !important; padding: 1rem; border-radius: 0.5rem; }
        .stButton>button { background-color: #2563eb; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.5rem; }
        .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Sales Prediction Dashboard")
st.markdown("""Use this interactive dashboard to:
- Upload your sales dataset
- Explore trends and feature correlations
- Train regression models
- Predict future sales and analyze outcomes
""")

# Upload dataset
uploaded_file = st.file_uploader("📁 Upload your sales dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    with st.expander("View Uploaded Data"):
        st.dataframe(df.head(), use_container_width=True)

    # Missing values
    st.subheader("🧽 Missing Values Summary")
    with st.expander("View Missing Data Details"):
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count", "index": "Column"}))

    # Select target column
    target_col = st.selectbox("🎯 Select the Target (Sales) Column", options=df.columns)
    feature_cols = st.multiselect("🧮 Select Feature Columns", options=[col for col in df.columns if col != target_col])

    if feature_cols:
        # Handle categorical features automatically
        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df[target_col]

        st.subheader("📈 Feature Correlation Heatmap")
        with st.expander("View Heatmap"):
            corr = df[[target_col] + feature_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=False, fmt=".2f", annot_kws={"size": 7})
            st.pyplot(fig)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_choice = st.selectbox("🤖 Choose Regression Model", ["Linear Regression", "Random Forest", "XGBoost"])

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.subheader("📊 Model Evaluation")
        col1, col2 = st.columns(2)
        col1.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

        # Plot actual vs predicted
        st.subheader("📉 Actual vs Predicted Sales")
        plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True)
        fig2 = px.line(plot_df, markers=True, title="Sales Comparison", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        # Optional: Predict on new data
        st.subheader("📥 Predict on New Input")
        with st.expander("Enter values for prediction"):
            input_data = {}
            for col in X.columns:
                val = st.number_input(f"{col}", value=0.0, format="%.2f")
                input_data[col] = val

            if st.button("📌 Predict Sales"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_df = input_df.copy()
                result_df["Predicted Sales"] = prediction
                result_df["Timestamp"] = timestamp
                st.success(f"✅ Predicted Sales: {prediction:.2f}")

                # Download prediction
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button("⬇️ Download Prediction Result", data=csv_buffer.getvalue(), file_name="sales_prediction.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to get started.")
