"""
Employee Burn Rate Prediction
Regression problem: Predict 'Burn Rate' for employees.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    return train, test


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
    df = df.copy()

    # Date features
    df["Date of Joining"] = pd.to_datetime(df["Date of Joining"], errors="coerce")
    df["join_year"] = df["Date of Joining"].dt.year
    df["join_month"] = df["Date of Joining"].dt.month
    df["join_day"] = df["Date of Joining"].dt.day
    df.drop(columns=["Date of Joining", "Employee ID"], inplace=True)

    # Fill missing numerics with median
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_col = "Burn Rate"
    if is_train and target_col in num_cols:
        num_cols.remove(target_col)
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill any remaining NaNs across all columns
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categoricals
    cat_cols = ["Gender", "Company Type", "WFH Setup Available"]
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = (
                df[col]
                .astype(str)
                .map(lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1)
            )

    return df, encoders


# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"  Validation RMSE : {rmse:.4f}")
    print(f"  Validation R²   : {r2:.4f}")
    return rmse, r2


# ─────────────────────────────────────────────
# 4. PREDICT & SAVE SUBMISSION
# ─────────────────────────────────────────────
def make_submission(model, test_processed, original_test, output_path: str):
    feature_cols = [c for c in test_processed.columns if c != "Burn Rate"]
    preds = model.predict(test_processed[feature_cols])
    submission = pd.DataFrame(
        {
            "Employee ID": original_test["Employee ID"],
            "Burn Rate": preds,
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved → {output_path}")
    print(submission.head())


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    OUTPUT_PATH = "submission.csv"

    # Load
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    original_test = test.copy()

    # Preprocess
    train_proc, encoders = preprocess(train, is_train=True)
    test_proc, _ = preprocess(test, is_train=False, encoders=encoders)

    # Split features / target
    TARGET = "Burn Rate"
    features = [c for c in train_proc.columns if c != TARGET]
    X = train_proc[features]
    y = train_proc[TARGET]

    print(f"\nFeatures used : {features}")
    print(f"Target        : {TARGET}\n")

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Train
    print("Training GradientBoostingRegressor …")
    model = train_model(X_train, y_train)

    # Evaluate
    print("\n── Evaluation ──")
    evaluate(model, X_val, y_val)

    # Cross-val for robustness
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_root_mean_squared_error"
    )
    print(f"  5-Fold CV RMSE  : {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Submission
    make_submission(model, test_proc, original_test, OUTPUT_PATH)


if __name__ == "__main__":
    main()
