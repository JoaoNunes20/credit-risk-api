"""Prediction API for the credit-risk model.

Run from the project root:
    uv run uvicorn credit_risk_api.api:app --reload
"""
import json
import logging
import time
from pathlib import Path
from typing import Literal, get_args

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from credit_risk_api import llm

MODEL_DIR = Path("models")
DEMO_PAGE = Path(__file__).parent / "static" / "index.html"

# One JSON line per prediction on stdout, so the platform's log viewer
# (or any log collector) can search and aggregate them.
logger = logging.getLogger("credit_risk_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

# Loaded once at startup, not on every request.
model = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())

ContractType = Literal["Cash loans", "Revolving loans"]
EducationType = Literal[
    "Academic degree",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Secondary / secondary special",
]
OccupationType = Literal[
    "Accountants",
    "Cleaning staff",
    "Cooking staff",
    "Core staff",
    "Drivers",
    "HR staff",
    "High skill tech staff",
    "IT staff",
    "Laborers",
    "Low-skill Laborers",
    "Managers",
    "Medicine staff",
    "Private service staff",
    "Realty agents",
    "Sales staff",
    "Secretaries",
    "Security staff",
    "Waiters/barmen staff",
]


class LoanApplication(BaseModel):
    """Input schema: one field per model feature, in lowercase."""

    ext_source_1: float | None = Field(None, ge=0, le=1, description="External credit score 1 (0 to 1)")
    ext_source_2: float | None = Field(None, ge=0, le=1, description="External credit score 2 (0 to 1)")
    ext_source_3: float | None = Field(None, ge=0, le=1, description="External credit score 3 (0 to 1)")
    amt_income_total: float = Field(gt=0, description="Annual income of the client")
    amt_credit: float = Field(gt=0, description="Credit amount of the loan")
    amt_annuity: float = Field(gt=0, description="Loan annuity (periodic payment)")
    amt_goods_price: float | None = Field(None, gt=0, description="Price of the goods the loan is for")
    days_employed: int | None = Field(
        None, le=0, description="Days employed before the application (negative); null if not employed, e.g. pensioners"
    )
    name_contract_type: ContractType
    name_education_type: EducationType
    occupation_type: OccupationType | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ext_source_1": 0.5,
                    "ext_source_2": 0.6,
                    "ext_source_3": 0.55,
                    "amt_income_total": 180000,
                    "amt_credit": 500000,
                    "amt_annuity": 25000,
                    "amt_goods_price": 450000,
                    "days_employed": -2000,
                    "name_contract_type": "Cash loans",
                    "name_education_type": "Higher education",
                    "occupation_type": "Laborers",
                }
            ]
        }
    }


class Prediction(BaseModel):
    default_probability: float


class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None
    contribution: float = Field(description="SHAP value in log-odds; positive increases default risk")


class Explanation(BaseModel):
    default_probability: float
    baseline_log_odds: float = Field(description="Model output before any feature is taken into account")
    contributions: list[FeatureContribution] = Field(description="Sorted by absolute impact, largest first")


class Source(BaseModel):
    ref: int = Field(description="Number used to cite this source in the narrative, e.g. [1]")
    document: str
    section: str
    page: int


class NarrativeExplanation(Explanation):
    narrative: str = Field(description="Plain-language explanation written by an LLM")
    sources: list[Source] = Field(description="Regulatory excerpts given to the LLM, cited by number")


def check_contract() -> None:
    """Fail at startup if the API schema does not match the trained model."""
    api_features = sorted(name.upper() for name in LoanApplication.model_fields)
    if api_features != sorted(metadata["features"]):
        raise RuntimeError("API fields do not match the model features in metadata.json")

    categories = {
        "NAME_CONTRACT_TYPE": ContractType,
        "NAME_EDUCATION_TYPE": EducationType,
        "OCCUPATION_TYPE": OccupationType,
    }
    for col, literal in categories.items():
        if list(get_args(literal)) != metadata["categorical"][col]:
            raise RuntimeError(f"API categories for {col} do not match metadata.json")


check_contract()

app = FastAPI(title="Credit Risk API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    """Interactive demo page; the API itself is documented at /docs."""
    return FileResponse(DEMO_PAGE)


@app.get("/model-info")
def model_info() -> dict:
    return {
        "features": metadata["features"],
        "excluded_features": metadata["excluded_features"],
        "test_auc": metadata["test_auc"],
    }


def to_model_input(application: LoanApplication) -> pd.DataFrame:
    """One-row DataFrame with the column names, order and dtypes used in training."""
    row = {name.upper(): value for name, value in application.model_dump().items()}
    X = pd.DataFrame([row], columns=metadata["features"])

    # Rebuild the exact dtypes used in training: categories and floats (None -> NaN).
    for col, categories in metadata["categorical"].items():
        X[col] = pd.Categorical(X[col], categories=categories)
    numeric = [col for col in metadata["features"] if col not in metadata["categorical"]]
    X[numeric] = X[numeric].astype("float64")
    return X


@app.post("/predict")
def predict(application: LoanApplication) -> Prediction:
    start = time.perf_counter()
    X = to_model_input(application)
    probability = round(float(model.predict_proba(X)[0, 1]), 4)

    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "default_probability": probability,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "inputs": application.model_dump(),
            }
        )
    )
    return Prediction(default_probability=probability)


def compute_explanation(application: LoanApplication) -> Explanation:
    """Prediction plus the contribution of each feature (SHAP values).

    LightGBM computes exact TreeSHAP values natively (pred_contrib=True), so no
    extra library is needed. Contributions are in log-odds and add up, together
    with the baseline, to the model's raw output for this application.
    """
    X = to_model_input(application)
    probability = round(float(model.predict_proba(X)[0, 1]), 4)
    shap_values = model.predict(X, pred_contrib=True)[0]  # one value per feature, then the baseline

    inputs = application.model_dump()
    contributions = [
        FeatureContribution(
            feature=col.lower(),
            value=inputs[col.lower()],
            contribution=round(float(shap_value), 4),
        )
        for col, shap_value in zip(metadata["features"], shap_values[:-1])
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    return Explanation(
        default_probability=probability,
        baseline_log_odds=round(float(shap_values[-1]), 4),
        contributions=contributions,
    )


@app.post("/explain")
def explain(application: LoanApplication) -> Explanation:
    """Prediction plus the contribution of each feature (SHAP values)."""
    return compute_explanation(application)


@app.post("/explain/narrative")
def explain_narrative(application: LoanApplication) -> NarrativeExplanation:
    """The /explain output plus a plain-language explanation written by an LLM."""
    if not llm.is_configured():
        raise HTTPException(status_code=503, detail="LLM explanations are not configured")
    explanation = compute_explanation(application)
    context = llm.regulatory_context(explanation.model_dump())
    try:
        narrative = llm.generate_narrative(explanation.model_dump(), context)
    except llm.LLMError as exc:
        logger.warning(json.dumps({"event": "llm_error", "error": str(exc)}))
        raise HTTPException(status_code=502, detail="The explanation service is unavailable") from exc
    sources = [
        Source(ref=number, document=chunk["source"], section=chunk["section"], page=chunk["page"])
        for number, chunk in enumerate(context, start=1)
    ]
    return NarrativeExplanation(**explanation.model_dump(), narrative=narrative, sources=sources)
