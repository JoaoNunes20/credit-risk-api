"""Tests for the prediction API.

Run from the project root (requires a trained model in models/):
    uv run pytest
"""
import json
import math

import pytest
from fastapi.testclient import TestClient

from credit_risk_api.api import app

client = TestClient(app)

VALID_APPLICATION = {
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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability():
    response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    probability = response.json()["default_probability"]
    assert 0 <= probability <= 1


def test_predict_accepts_missing_optional_fields():
    application = {
        **VALID_APPLICATION,
        "ext_source_1": None,
        "occupation_type": None,
        "days_employed": None,  # e.g. a pensioner
    }
    response = client.post("/predict", json=application)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("name_contract_type", "Mortgage"),  # unknown category
        ("ext_source_2", 1.7),  # score above 1
        ("amt_credit", -1000),  # negative amount
        ("days_employed", 500),  # days employed must be negative
    ],
)
def test_predict_rejects_invalid_values(field, value):
    response = client.post("/predict", json={**VALID_APPLICATION, field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_predict_rejects_missing_required_field():
    application = {k: v for k, v in VALID_APPLICATION.items() if k != "amt_credit"}
    response = client.post("/predict", json=application)
    assert response.status_code == 422


def test_sensitive_attributes_are_not_model_inputs():
    info = client.get("/model-info").json()
    for attribute in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"):
        assert attribute not in info["features"]
        assert attribute in info["excluded_features"]


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_prediction_is_logged(caplog):
    caplog.set_level("INFO", logger="credit_risk_api")
    response = client.post("/predict", json=VALID_APPLICATION)

    records = [r for r in caplog.records if r.name == "credit_risk_api"]
    assert len(records) == 1
    entry = json.loads(records[0].getMessage())
    assert entry["event"] == "prediction"
    assert entry["default_probability"] == response.json()["default_probability"]
    assert entry["inputs"]["amt_credit"] == VALID_APPLICATION["amt_credit"]


def test_explain_contributions_add_up_to_the_prediction():
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    log_odds = explanation["baseline_log_odds"] + sum(
        c["contribution"] for c in explanation["contributions"]
    )
    probability = 1 / (1 + math.exp(-log_odds))
    assert probability == pytest.approx(explanation["default_probability"], abs=1e-3)


def test_explain_covers_every_feature_sorted_by_impact():
    contributions = client.post("/explain", json=VALID_APPLICATION).json()["contributions"]
    assert {c["feature"] for c in contributions} == set(VALID_APPLICATION)
    impacts = [abs(c["contribution"]) for c in contributions]
    assert impacts == sorted(impacts, reverse=True)
