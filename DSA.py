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

# =============================
# --- Page Setup
# =============================
st.set_page_config(page_title="Smart Share Market Prediction", page_icon="📊", layout="wide")
st.title("🤖 Smart Share Market Prediction (Fully Auto & Error-Free)")

# =============================
# --- Load Dataset
# =============================
DATA_URL = "https://drive.google.com/uc?export=download&id=1006n43OyDiOzLsKH-deZS-HOi4P6KnbS"

try:
    st.subheader("📥 Loading Dataset...")
    response = requests.get(DATA_URL, allow_redirects=True, timeout=25)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    st.success("✅ Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"⚠️ Failed to load dataset: {e}")
    st.stop()

# =============================
# --- Select Target Variable
# =============================
all_cols = df.columns.tolist()
target = st.selectbox("🎯 Select Target Variable", all_cols, index=len(all_cols) - 1)

# --- Drop NaN rows where target missing ---
df = df.dropna(subset=[target])

# --- Separate Features & Target ---
X = df.drop(columns=[target])
y = df[target]

# --- Handle NaN in target (safety) ---
if y.isna().sum() > 0:
    y = y.fillna(y.mean())

# =============================
# --- Identify Numeric & Categorical Columns
# =============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# =============================
# --- Preprocessing Pipelines
# =============================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =============================
# --- Define Model Pipelines
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
# --- Train & Evaluate Models
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
        st.warning(f"⚠️ {name} failed: {e}")

# =============================
# --- Auto-Select Best Model
# =============================
if not performance:
    st.error("❌ No model could be evaluated successfully.")
    st.stop()

best_model_name = max(performance, key=lambda k: performance[k]["r2"])
best_model = performance[best_model_name]["model"]
best_r2 = performance[best_model_name]["r2"]

st.success(f"🏆 Best Model Selected Automatically: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# --- Plot Actual vs Predicted
# =============================
try:
    y_pred_best = best_model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred_best, color='blue', alpha=0.6, label='Predicted')
    ax.plot(y, y, color='red', label='Actual')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs Predicted ({best_model_name})")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Plotting failed: {e}")

# =============================
# --- Manual Input Prediction
# =============================
st.markdown("---")
st.subheader("🧮 Try Your Own Input")

user_input = {}
cols = st.columns(2)
for i, col_name in enumerate(X.columns):
    with cols[i % 2]:
        sample_val = df[col_name].iloc[0]
        if isinstance(sample_val, (int, float)):
            user_input[col_name] = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
        else:
            user_input[col_name] = st.text_input(f"{col_name}", value=str(sample_val))

if st.button("🔮 Predict Automatically"):
    input_df = pd.DataFrame([user_input])
    prediction = best_model.predict(input_df)[0]
    st.success(f"📈 Predicted {target}: {prediction:.2f}")
    st.info(f"🤖 Model Used: **{best_model_name}** (R² = {best_r2:.4f})")

# =============================
# --- Model Comparison Table
# =============================
st.markdown("---")
st.subheader("📊 Model Performance Comparison")

perf_df = pd.DataFrame({
    "Model": performance.keys(),
    "R² Score": [round(v["r2"], 4) for v in performance.values()],
    "MSE": [round(v["mse"], 2) for v in performance.values()]
}).sort_values(by="R² Score", ascending=False)

st.table(perf_df)

# =============================
# --- Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <p>Developed with ❤️ by <b>Sumiya Ahasan</b></p>
        <p style='font-size:13px;'>© 2025 Smart Share Market ML App | Fully Auto Pipeline Version</p>
    </div>
    """,
    unsafe_allow_html=True
)
