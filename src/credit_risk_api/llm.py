"""Natural-language explanations of predictions with an LLM.

Works with any OpenAI-compatible chat API, configured with environment variables:
- LLM_API_KEY:  required; without it the narrative endpoint is disabled
- LLM_BASE_URL: optional, defaults to Groq
- LLM_MODEL:    optional, defaults to an open-weight model available on Groq's free tier

Switching provider only means changing LLM_BASE_URL and LLM_MODEL.
"""
import math
import os

from openai import OpenAI, OpenAIError

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Readable names for the model features, so the LLM never sees column codes.
FEATURE_DESCRIPTIONS = {
    "ext_source_1": "external credit score 1 (scale 0 to 1)",
    "ext_source_2": "external credit score 2 (scale 0 to 1)",
    "ext_source_3": "external credit score 3 (scale 0 to 1)",
    "amt_income_total": "annual income",
    "amt_credit": "loan amount",
    "amt_annuity": "periodic loan payment",
    "amt_goods_price": "price of the goods being financed",
    "days_employed": "time in current employment",
    "name_contract_type": "type of loan",
    "name_education_type": "education level",
    "occupation_type": "occupation",
}

AMOUNT_FEATURES = {"amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price"}

SYSTEM_PROMPT = """You explain the output of a credit risk model to loan officers.

Rules:
- Use only the factors, values and impact labels you are given. Never invent
  factors, data or reasons.
- Describe each factor's impact with the label provided (strong, moderate, small,
  negligible). Do not judge values yourself: never call a value high, low, good,
  bad, short or long.
- Amounts are in the loan's currency. Do not add currency symbols.
- Explain which factors increased and which decreased the estimated risk,
  strongest first. You may leave out factors with negligible impact.
- Use plain language. Do not mention SHAP, log-odds or other technical terms.
- The estimate supports a human decision; it is not a decision. Never say whether
  the loan should be approved or rejected.
- End with the estimated probability of default compared with the baseline
  for a typical applicant, using the percentages given.
- Write 4 to 6 sentences in English."""


class LLMError(Exception):
    """The LLM provider could not produce an explanation."""


def is_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def impact_label(contribution: float) -> str:
    """Fixed thresholds on the SHAP value (log-odds), so the code, not the LLM, judges size."""
    size = abs(contribution)
    if size >= 0.15:
        return "strong"
    if size >= 0.05:
        return "moderate"
    if size >= 0.02:
        return "small"
    return "negligible"


def format_value(feature: str, value) -> str:
    """Human-readable value: years instead of negative days, thousands separators."""
    if value is None:
        if feature == "days_employed":
            return "not employed (e.g. retired)"
        return "not available"
    if feature == "days_employed":
        return f"{-value / 365:.1f} years"
    if feature in AMOUNT_FEATURES:
        return f"{value:,.0f}"
    return str(value)


def build_prompt(explanation: dict) -> str:
    """Turn the /explain output into a readable summary for the LLM."""
    probability = explanation["default_probability"]
    baseline = 1 / (1 + math.exp(-explanation["baseline_log_odds"]))
    lines = [
        f"Estimated probability of default for this applicant: {probability:.1%}",
        f"Baseline for a typical applicant: {baseline:.1%}",
        "",
        "Factors, strongest first:",
    ]
    for c in explanation["contributions"]:
        direction = "increases risk" if c["contribution"] > 0 else "decreases risk"
        lines.append(
            f"- {FEATURE_DESCRIPTIONS[c['feature']]}: {format_value(c['feature'], c['value'])} "
            f"-> {direction} ({impact_label(c['contribution'])} impact)"
        )
    return "\n".join(lines)


def generate_narrative(explanation: dict) -> str:
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        timeout=20,
        max_retries=1,
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(explanation)},
            ],
            temperature=0.2,  # low: we want faithful, consistent explanations
            max_tokens=1000,  # leaves room for models that reason before answering
        )
    except OpenAIError as exc:
        raise LLMError(str(exc)) from exc
    narrative = (response.choices[0].message.content or "").strip()
    if not narrative:
        raise LLMError("The model returned an empty response")
    return narrative
