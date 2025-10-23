import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="Smart Share Market Prediction", page_icon="📊", layout="wide")
st.title("🤖 Smart Share Market Prediction (Debug Mode)")

# =============================
# --- Load Dataset
# =============================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

try:
    response = requests.get(DATA_URL, allow_redirects=True, timeout=25)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.exception(e)
    st.stop()

# =============================
# --- Select Target
# =============================
all_cols = df.columns.tolist()
target = st.selectbox("🎯 Select Target Variable", all_cols, index=len(all_cols) - 1)

try:
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]
except Exception as e:
    st.exception(e)
    st.stop()

# =============================
# --- Preprocessing
# =============================
try:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
except Exception as e:
    st.exception(e)
    st.stop()

# =============================
# --- Models
# =============================
pipelines = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "XGBoost Regressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(random_state=42, n_estimators=200, verbosity=0))
    ]),
    "Support Vector Machine (SVM)": Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_estimators=150))
    ])
}

# =============================
# --- Train & Evaluate
# =============================
st.subheader("🧠 Training and Evaluating Models...")

performance = {}
for name, pipeline in pipelines.items():
    try:
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        performance[name] = {"model": pipeline, "r2": r2, "mse": mse}
        st.write(f"✅ {name}: R² = {r2:.4f}, MSE = {mse:.2f}")
    except Exception as e:
        st.warning(f"⚠️ {name} failed:")
        st.exception(e)

if not performance:
    st.error("❌ No model trained successfully.")
    st.stop()

best_model_name = max(performance, key=lambda k: performance[k]["r2"])
best_model = performance[best_model_name]["model"]
best_r2 = performance[best_model_name]["r2"]

st.success(f"🏆 Best Model: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# --- Prediction Plot
# =============================
try:
    y_pred_best = best_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred_best, color="blue", alpha=0.6)
    ax.plot(y, y, color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
except Exception as e:
    st.exception(e)

# =============================
# --- Manual Input
# =============================
st.markdown("---")
st.subheader("🧮 Manual Prediction")

try:
    user_input = {}
    cols = st.columns(2)
    for i, col_name in enumerate(X.columns):
        with cols[i % 2]:
            sample_val = df[col_name].iloc[0]
            if isinstance(sample_val, (int, float)):
                user_input[col_name] = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
            else:
                user_input[col_name] = st.text_input(f"{col_name}", value=str(sample_val))

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([user_input])
        pred_value = best_model.predict(input_df)[0]
        st.success(f"📈 Predicted {target}: {pred_value:.2f}")
        st.info(f"🤖 Model Used: {best_model_name} (R² = {best_r2:.4f})")
except Exception as e:
    st.exception(e)
