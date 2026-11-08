"""Tests for the prediction API.

Run from the project root (requires a trained model in models/):
    uv run pytest
"""
import json
import math

import pytest
from fastapi.testclient import TestClient

from credit_risk_api import llm
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


def test_root_serves_the_demo_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Estimate risk" in response.text


def test_demo_page_select_options_come_from_the_schema():
    # The page fills its dropdowns from /openapi.json; check the enums are there.
    props = client.get("/openapi.json").json()["components"]["schemas"]["LoanApplication"]["properties"]
    assert "Cash loans" in props["name_contract_type"]["enum"]
    occupations = [v for s in props["occupation_type"]["anyOf"] for v in s.get("enum", [])]
    assert "Laborers" in occupations


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


def test_narrative_returns_llm_text(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm, "generate_narrative", lambda explanation, context: "Mocked explanation.")
    response = client.post("/explain/narrative", json=VALID_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert body["narrative"] == "Mocked explanation."
    assert len(body["contributions"]) == len(VALID_APPLICATION)
    assert [s["ref"] for s in body["sources"]] == list(range(1, len(body["sources"]) + 1))


def test_narrative_without_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    response = client.post("/explain/narrative", json=VALID_APPLICATION)
    assert response.status_code == 503


def test_narrative_provider_error_returns_502(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def failing_llm(explanation, context):
        raise llm.LLMError("provider down")

    monkeypatch.setattr(llm, "generate_narrative", failing_llm)
    response = client.post("/explain/narrative", json=VALID_APPLICATION)
    assert response.status_code == 502


def test_prompt_uses_readable_names_and_the_probability():
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    prompt = llm.build_prompt(explanation)
    assert f"{explanation['default_probability']:.1%}" in prompt
    assert "external credit score" in prompt
    assert "ext_source" not in prompt  # the LLM never sees raw column codes


def test_prompt_turns_values_into_neutral_readable_facts():
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    prompt = llm.build_prompt(explanation)
    assert "5.5 years" in prompt  # -2000 days, not a negative number
    assert "500,000" in prompt  # thousands separator, no currency symbol
    assert "$" not in prompt


def test_impact_labels_use_fixed_thresholds():
    assert llm.impact_label(-0.2276) == "strong"
    assert llm.impact_label(0.0836) == "moderate"
    assert llm.impact_label(0.0264) == "small"
    assert llm.impact_label(-0.005) == "negligible"


def test_regulatory_context_covers_strong_factors_and_automated_models():
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    context = llm.regulatory_context(explanation)
    assert 1 <= len(context) <= llm.MAX_TOPICS_WITH_CONTEXT + 1
    assert len({c["id"] for c in context}) == len(context)  # no duplicates
    _, retrieval = llm.RAG_INDEX
    model_rules = retrieval["topics"]["automated_models"][0]["chunk_id"]
    assert model_rules in {c["id"] for c in context}


def test_prompt_numbers_the_regulatory_excerpts():
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    context = llm.regulatory_context(explanation)
    prompt = llm.build_prompt(explanation, context)
    assert "Regulatory excerpts:" in prompt
    assert f"[{len(context)}]" in prompt


def test_regulatory_context_uses_distinct_topics():
    _, retrieval = llm.RAG_INDEX
    explanation = {
        "contributions": [
            {"feature": "name_education_type", "contribution": -0.23},
            {"feature": "ext_source_3", "contribution": -0.19},
            {"feature": "ext_source_2", "contribution": -0.16},  # same topic as ext_source_3
            {"feature": "amt_credit", "contribution": 0.14},
            {"feature": "ext_source_1", "contribution": -0.13},
        ]
    }
    expected_topics = ["personal_data", "external_scores", "loan_amount", "automated_models"]
    expected_ids = []
    for topic in expected_topics:
        chunk_id = retrieval["topics"][topic][0]["chunk_id"]
        if chunk_id not in expected_ids:
            expected_ids.append(chunk_id)
    assert [c["id"] for c in llm.regulatory_context(explanation)] == expected_ids


def test_find_forbidden_phrases_flags_judgements_and_compliance_claims():
    text = "Higher education and a score of 0.55 lower the risk slightly; a relatively large loan raises it."
    assert llm.find_forbidden_phrases(text) == ["large", "relatively"]
    assert llm.find_forbidden_phrases("The very good score of 0.6 strongly lowers risk.") == ["very good"]
    assert llm.find_forbidden_phrases("This explanation follows the EBA guidance [1].") == ["follows"]
    assert llm.find_forbidden_phrases("The guidelines ask lenders to verify information [1].") == []


def test_narrative_with_judgements_is_rewritten_once(monkeypatch):
    replies = iter(["A very good score lowers risk.", "A score of 0.6 lowers risk."])
    calls = []

    def fake_complete(messages):
        calls.append(messages)
        return next(replies)

    monkeypatch.setattr(llm, "_complete", fake_complete)
    explanation = client.post("/explain", json=VALID_APPLICATION).json()
    assert llm.generate_narrative(explanation) == "A score of 0.6 lowers risk."
    assert len(calls) == 2
    assert "very good" in calls[1][-1]["content"]
