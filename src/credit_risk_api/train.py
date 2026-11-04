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

from credit_risk_api.drift import build_reference

DATA_PATH = Path("data/raw/application_train.csv")
MODEL_DIR = Path("models")

# Feature set chosen in notebooks/03_information_value.ipynb:
# the original baseline minus sensitive attributes, plus OCCUPATION_TYPE.
NUMERIC_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
]
CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "TARGET"

# Attributes deliberately left out of the model, with the reason for each.
EXCLUDED_FEATURES = {
    "CODE_GENDER": "sex: prohibited in EU credit decisions (Directive 2004/113/EC)",
    "DAYS_BIRTH": "age: protected characteristic, excluded as a precaution",
    "NAME_FAMILY_STATUS": "family status: sensitive, excluded as a precaution",
    "CNT_CHILDREN": "family status proxy",
    "CNT_FAM_MEMBERS": "family status proxy",
    "REGION_RATING_CLIENT": "geography: potential proxy for protected groups",
    "REGION_RATING_CLIENT_W_CITY": "geography: potential proxy for protected groups",
    "REGION_POPULATION_RELATIVE": "geography: potential proxy for protected groups",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=FEATURES + [TARGET])
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
        "excluded_features": EXCLUDED_FEATURES,
        "test_auc": round(auc, 4),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Training distribution of each feature, the baseline for drift monitoring.
    reference = build_reference(X_train, CATEGORICAL_FEATURES)
    (MODEL_DIR / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Saved model, metadata and drift reference to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
