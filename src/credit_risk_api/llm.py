"""Natural-language explanations of predictions with an LLM.

Works with any OpenAI-compatible chat API, configured with environment variables:
- LLM_API_KEY:  required; without it the narrative endpoint is disabled
- LLM_BASE_URL: optional, defaults to Groq
- LLM_MODEL:    optional, defaults to an open-weight model available on Groq's free tier

Switching provider only means changing LLM_BASE_URL and LLM_MODEL.
"""
import json
import logging
import math
import os
import re
from pathlib import Path

from openai import OpenAI, OpenAIError

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

RAG_DIR = Path("models/rag")
MAX_TOPICS_WITH_CONTEXT = 3  # regulatory excerpts for the strongest factors' topics only

logger = logging.getLogger("credit_risk_api")

# Phrases the prompt forbids: value judgements and claims of regulatory compliance.
# Prompts are instructions, not guarantees, so the output is checked in code.
# Word boundaries keep "higher education" and "lower than" valid.
FORBIDDEN_PHRASES = re.compile(
    r"\b(very good|good|bad|poor|excellent|high|low|large|short|long|relatively"
    r"|follows|complies|compliant|compliance|in line with|in accordance with)\b",
    re.IGNORECASE,
)

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
- After the factors, state the estimated probability of default compared with the
  baseline for a typical applicant, using the percentages given.
- If regulatory excerpts are provided, finish with one or two sentences relating the
  explanation to them, citing them as [1], [2], etc. Only state what the excerpts
  actually say. If none is relevant, do not mention regulation.
- Never claim that the applicant, the model or this explanation complies with any
  regulation.
- Write one flowing paragraph without line breaks or lists. Express the impact labels
  naturally (e.g. 'strongly', 'slightly') instead of copying them in brackets.
- Write 5 to 8 sentences in English."""


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


def load_rag_index() -> tuple[list[dict], dict] | None:
    """Chunks and precomputed retrieval built by build_rag_index, if present."""
    try:
        chunks = json.loads((RAG_DIR / "chunks.json").read_text())
        retrieval = json.loads((RAG_DIR / "retrieval.json").read_text())
    except FileNotFoundError:
        return None
    return chunks, retrieval


RAG_INDEX = load_rag_index()


def regulatory_context(explanation: dict) -> list[dict]:
    """Regulatory excerpts for the strongest factors, plus the rules on automated models.

    Retrieval was done offline for every topic and reviewed by hand; here we only pick
    the top excerpt of each relevant topic, without duplicates.
    """
    if RAG_INDEX is None:
        return []
    chunks, retrieval = RAG_INDEX
    topics = []
    for c in explanation["contributions"]:  # sorted by impact, strongest first
        topic = retrieval["feature_topics"][c["feature"]]
        if impact_label(c["contribution"]) != "negligible" and topic not in topics:
            topics.append(topic)  # distinct topics: three external scores count once
        if len(topics) == MAX_TOPICS_WITH_CONTEXT:
            break
    topics.append("automated_models")

    context, seen = [], set()
    for topic in topics:
        chunk_id = retrieval["topics"][topic][0]["chunk_id"]
        if chunk_id not in seen:
            seen.add(chunk_id)
            context.append(chunks[chunk_id])
    return context


def build_prompt(explanation: dict, context: list[dict] = ()) -> str:
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
    if context:
        lines += ["", "Regulatory excerpts:"]
        for number, chunk in enumerate(context, start=1):
            lines.append(f"[{number}] {chunk['source']}, {chunk['section']}, p. {chunk['page']}: {chunk['text']}")
    return "\n".join(lines)


def find_forbidden_phrases(text: str) -> list[str]:
    """Forbidden phrases found in a narrative, e.g. ['follows', 'relatively']."""
    return sorted({match.group(0).lower() for match in FORBIDDEN_PHRASES.finditer(text)})


def _complete(messages: list[dict]) -> str:
    """One chat completion from the configured provider."""
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        timeout=20,
        max_retries=1,
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            messages=messages,
            temperature=0.2,  # low: we want faithful, consistent explanations
            max_tokens=1000,  # leaves room for models that reason before answering
        )
    except OpenAIError as exc:
        raise LLMError(str(exc)) from exc
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("The model returned an empty response")
    return text


def generate_narrative(explanation: dict, context: list[dict] = ()) -> str:
    """Narrative explanation, rewritten once if it contains forbidden phrases."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(explanation, context)},
    ]
    narrative = _complete(messages)

    forbidden = find_forbidden_phrases(narrative)
    if forbidden:
        messages += [
            {"role": "assistant", "content": narrative},
            {
                "role": "user",
                "content": "Rewrite the paragraph without these words: "
                + ", ".join(forbidden)
                + ". They are value judgements or claims of regulatory compliance. Describe values"
                " neutrally, describe regulation only by what the excerpts say, and keep"
                " everything else.",
            },
        ]
        narrative = _complete(messages)
        remaining = find_forbidden_phrases(narrative)
        logger.warning(
            json.dumps({"event": "narrative_rewritten", "forbidden": forbidden, "remaining": remaining})
        )
    return narrative
