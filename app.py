import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Burn Rate Predictor", page_icon="🔥", layout="wide"
)
st.title("🔥 Employee Burn Rate Predictor")
st.markdown(
    "Predict how likely an employee is to burn out — and act before it happens."
)


# ── Load & Preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    def preprocess(df):
        df = df.copy()
        df["Date of Joining"] = pd.to_datetime(df["Date of Joining"], errors="coerce")
        df["Months_Since_Joining"] = (
            (pd.Timestamp("2021-01-01") - df["Date of Joining"]).dt.days / 30
        ).round(1)
        df.drop(columns=["Employee ID", "Date of Joining"], inplace=True)
        for col in ["Gender", "Company Type", "WFH Setup Available"]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df["Fatigue_x_Resource"] = (
            df["Mental Fatigue Score"] * df["Resource Allocation"]
        )
        df["Designation_x_Fatigue"] = df["Designation"] * df["Mental Fatigue Score"]
        return df

    train_p = preprocess(train)
    test_p = preprocess(test)

    FEATURES = [
        "Gender",
        "Company Type",
        "WFH Setup Available",
        "Designation",
        "Resource Allocation",
        "Mental Fatigue Score",
        "Months_Since_Joining",
        "Fatigue_x_Resource",
        "Designation_x_Fatigue",
    ]
    train_clean = train_p.dropna(subset=FEATURES + ["Burn Rate"])
    X = train_clean[FEATURES]
    y = train_clean["Burn Rate"]
    X_test = test_p[FEATURES].fillna(test_p[FEATURES].median())
    return train, train_clean, X, y, X_test, test, FEATURES


@st.cache_data
def train_models(X, y):
    models = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    results = {}
    for name, model in models.items():
        cv_rmse = -cross_val_score(
            model, X, y, cv=5, scoring="neg_root_mean_squared_error"
        )
        model.fit(X, y)
        results[name] = {
            "model": model,
            "cv_rmse": cv_rmse.mean(),
            "cv_std": cv_rmse.std(),
        }
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Data Source")
use_demo = st.sidebar.checkbox("Use local train.csv / test.csv", value=True)

if use_demo:
    TRAIN_PATH, TEST_PATH = "train.csv", "test.csv"
    st.sidebar.info("Place train.csv and test.csv in the same folder as app.py")
else:
    train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
    test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")
    if train_file and test_file:
        TRAIN_PATH, TEST_PATH = train_file, test_file
    else:
        st.sidebar.warning("Please upload both files.")
        st.stop()

train_raw, train_clean, X, y, X_test, test_raw, FEATURES = load_and_preprocess(
    TRAIN_PATH, TEST_PATH
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 EDA", "🤖 Model", "📈 Predictions", "🧑 Single Prediction"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Rows", f"{len(train_raw):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing Burn Rate", f"{train_raw['Burn Rate'].isna().sum():,}")
    st.dataframe(train_raw.head(10), use_container_width=True)

    st.subheader("Burn Rate Distribution")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(
        train_clean["Burn Rate"],
        bins=40,
        color="#e74c3c",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("Burn Rate")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close()

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        train_clean.select_dtypes(include=np.number).corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax2,
    )
    st.pyplot(fig2)
    plt.close()

    st.subheader("Avg Burn Rate by Category")
    cat_col = st.selectbox("Feature", ["Gender", "Company Type", "WFH Setup Available"])
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    train_raw.groupby(cat_col)["Burn Rate"].mean().plot(
        kind="bar", ax=ax3, color=["#3498db", "#e74c3c"]
    )
    ax3.set_ylabel("Mean Burn Rate")
    ax3.tick_params(axis="x", rotation=0)
    st.pyplot(fig3)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 2 – Model Training
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Comparison (5-Fold CV RMSE)")
    with st.spinner("Training models…"):
        results = train_models(X, y)

    rows = [
        {"Model": n, "CV RMSE": round(r["cv_rmse"], 5), "Std": round(r["cv_std"], 5)}
        for n, r in results.items()
    ]
    df_res = pd.DataFrame(rows).sort_values("CV RMSE")
    st.dataframe(df_res.set_index("Model"), use_container_width=True)

    best_name = df_res.iloc[0]["Model"]
    best_model = results[best_name]["model"]
    st.success(f"✅ Best model: **{best_name}** — RMSE = {df_res.iloc[0]['CV RMSE']}")

    st.subheader("Feature Importances")
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        fi.plot(kind="barh", ax=ax4, color="#2ecc71")
        ax4.set_title(f"Feature Importances — {best_name}")
        st.pyplot(fig4)
        plt.close()
    else:
        coef = pd.Series(np.abs(best_model.coef_), index=FEATURES).sort_values()
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        coef.plot(kind="barh", ax=ax4, color="#9b59b6")
        ax4.set_title(f"|Coefficients| — {best_name}")
        st.pyplot(fig4)
        plt.close()

    st.subheader("Actual vs Predicted (Train Set)")
    y_pred_train = best_model.predict(X)
    fig5, ax5 = plt.subplots(figsize=(5, 5))
    ax5.scatter(y, y_pred_train, alpha=0.3, s=5, color="#3498db")
    ax5.plot([0, 1], [0, 1], "r--", lw=1.5)
    ax5.set_xlabel("Actual")
    ax5.set_ylabel("Predicted")
    rmse_t = np.sqrt(mean_squared_error(y, y_pred_train))
    r2_t = r2_score(y, y_pred_train)
    ax5.text(0.05, 0.9, f"RMSE={rmse_t:.4f}  R²={r2_t:.4f}", transform=ax5.transAxes)
    st.pyplot(fig5)
    plt.close()

# ════════════════════════════════════════════════════════════
# TAB 3 – Test Set Predictions
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predictions on Test Set")
    best_name2 = min(results, key=lambda k: results[k]["cv_rmse"])
    best_model2 = results[best_name2]["model"]
    preds = best_model2.predict(X_test)

    submission = pd.DataFrame(
        {
            "Employee ID": test_raw["Employee ID"],
            "Burn Rate": preds.round(4),
        }
    )
    st.dataframe(submission.head(20), use_container_width=True)

    ca, cb = st.columns(2)
    ca.metric("Avg Predicted Burn Rate", f"{preds.mean():.3f}")
    cb.metric("High-Risk Employees (≥ 0.7)", f"{(preds >= 0.7).sum():,}")

    fig6, ax6 = plt.subplots(figsize=(8, 3))
    ax6.hist(preds, bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
    ax6.axvline(0.7, color="red", linestyle="--", label="Risk threshold (0.7)")
    ax6.legend()
    ax6.set_xlabel("Predicted Burn Rate")
    ax6.set_ylabel("Count")
    st.pyplot(fig6)
    plt.close()

    csv_out = submission.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download submission.csv", csv_out, "submission.csv", "text/csv"
    )

# ════════════════════════════════════════════════════════════
# TAB 4 – Single Employee Prediction
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predict Burn Rate for One Employee")

    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    company = c2.selectbox("Company Type", ["Service", "Product"])
    wfh = c3.selectbox("WFH Setup Available", ["Yes", "No"])

    c4, c5, c6 = st.columns(3)
    designation = c4.slider("Designation (0=Junior … 5=Senior)", 0, 5, 2)
    resource = c5.slider("Resource Allocation (hrs/day)", 1, 10, 5)
    fatigue = c6.slider("Mental Fatigue Score (0–10)", 0.0, 10.0, 5.0, step=0.1)
    months = st.slider("Months Since Joining", 1, 120, 36)

    input_df = pd.DataFrame(
        [
            {
                "Gender": 1 if gender == "Male" else 0,
                "Company Type": 1 if company == "Service" else 0,
                "WFH Setup Available": 1 if wfh == "Yes" else 0,
                "Designation": designation,
                "Resource Allocation": resource,
                "Mental Fatigue Score": fatigue,
                "Months_Since_Joining": months,
                "Fatigue_x_Resource": fatigue * resource,
                "Designation_x_Fatigue": designation * fatigue,
            }
        ]
    )

    best_single = min(results, key=lambda k: results[k]["cv_rmse"])
    pred = results[best_single]["model"].predict(input_df)[0]

    color = "#e74c3c" if pred >= 0.7 else "#f39c12" if pred >= 0.5 else "#2ecc71"
    risk = (
        "🔴 High Risk"
        if pred >= 0.7
        else "🟡 Moderate Risk" if pred >= 0.5 else "🟢 Low Risk"
    )

    st.markdown(
        f"""
    <div style="background:{color}22; border-left:6px solid {color};
                padding:1.2rem; border-radius:8px; margin-top:1rem;">
        <h2 style="color:{color}; margin:0;">Burn Rate: {pred:.3f}</h2>
        <p style="font-size:1.1rem; margin:0.3rem 0 0;">{risk}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Thresholds: < 0.5 = Low Risk · 0.5–0.7 = Moderate · ≥ 0.7 = High Risk")
