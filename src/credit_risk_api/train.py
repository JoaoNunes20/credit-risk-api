"""Train a baseline credit-risk model and save it for the API.

Run from the project root:
    uv run python -m credit_risk_api.train
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/raw/application_train.csv")
MODEL_DIR = Path("models")

# A small, readable feature set keeps the API input simple.
NUMERIC_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
CATEGORICAL_FEATURES = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "TARGET"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])
    df = df[df["CODE_GENDER"] != "XNA"]  # only 4 rows; the API accepts F/M only
    # 365243 is a placeholder for "not employed" (pensioners): treat as missing.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].mask(df["DAYS_EMPLOYED"] == 365243)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def main() -> None:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Test AUC: {auc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    # The API needs the exact feature order and category values used in training.
    metadata = {
        "features": FEATURES,
        "categorical": {
            col: list(X_train[col].cat.categories) for col in CATEGORICAL_FEATURES
        },
        "test_auc": round(auc, 4),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
