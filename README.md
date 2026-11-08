# Credit Risk API

[![CI](https://github.com/JoaoNunes20/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/JoaoNunes20/credit-risk-api/actions/workflows/ci.yml)

An end-to-end machine learning project that takes a credit default model from
training to a deployed, containerized prediction service.

The focus is on **ML engineering**: serving, validation, testing, packaging,
continuous integration, deployment, monitoring and explainability, including
LLM-written explanations grounded in EU credit regulation. The model is a deliberately simple baseline; the goal
is to build everything around it the way it would be built in production.

**Live API:** https://credit-risk-api-uo5r.onrender.com (interactive docs)

**Technical report:** [docs/technical-report.md](docs/technical-report.md) ([PDF](docs/technical-report.pdf))

> Hosted on Render's free tier: after 15 minutes without traffic the service
> sleeps, so the first request can take up to a minute while it wakes up.

> **Status:** all six phases of the [roadmap](#roadmap) are done.

## The problem

Given a loan application, predict the probability that the client will have
payment difficulties. The data comes from the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
Kaggle competition: 307,511 applications, of which about 8% defaulted.

Because the classes are imbalanced, the model is evaluated with **ROC AUC**
rather than accuracy (predicting "no default" for everyone would already be
92% accurate).

## The model

A LightGBM classifier trained on 11 features:

| Type | Features |
|---|---|
| External credit scores | `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Loan and income amounts | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Employment | `DAYS_EMPLOYED`, `OCCUPATION_TYPE` |
| Other | `NAME_CONTRACT_TYPE`, `NAME_EDUCATION_TYPE` |

**Test ROC AUC: 0.750** on a stratified 80/20 split.

### Feature selection

Features were selected by combining Information Value (IV) analysis with
model-based comparisons ([notebook 03](notebooks/03_information_value.ipynb)).
Two findings shaped the final set:

- **IV alone underperformed.** The top 12 features by IV scored lower than the
  hand-picked baseline. IV evaluates each feature in isolation, so it misses
  interactions such as the relationship between the loan amount, the annuity
  and the price of the goods, which a tree-based model exploits.
- **High-cardinality categoricals can hurt.** `ORGANIZATION_TYPE` (58
  categories) had a high IV but lowered test AUC when added to the model.

### Excluded attributes

Sensitive attributes are excluded from the model by design, even where they
carried predictive signal:

| Attribute | Reason |
|---|---|
| `CODE_GENDER` | Sex may not be used in EU credit decisions (Directive 2004/113/EC) |
| `DAYS_BIRTH` | Age is a protected characteristic; excluded as a precaution |
| Family status (`NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`) | Sensitive; excluded as a precaution |
| Region (`REGION_RATING_CLIENT*`, `REGION_POPULATION_RELATIVE`) | Potential proxies for protected groups |

Removing these cost roughly 0.005 AUC. The list, with reasons, lives in
`EXCLUDED_FEATURES` in the training script, is saved alongside the model, is
exposed by the API's `/model-info` endpoint and is enforced by a test.

### Known limitations

- **Proxies.** Removing a sensitive attribute does not guarantee the model
  ignores it. `OCCUPATION_TYPE` is correlated with gender for some
  occupations, and the external scores (`EXT_SOURCE_*`) are produced by third
  parties that may use age or other sensitive data. A bias audit across
  protected groups is planned.
- **Single split.** Model comparisons used one train/test split. Differences
  of around 0.002 AUC are within the noise of a single split; cross-validation
  would give more reliable comparisons.

## The API

A [FastAPI](https://fastapi.tiangolo.com/) service with five endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (used by the platform's health checks) |
| `GET` | `/model-info` | Model features, excluded attributes and test AUC |
| `POST` | `/predict` | Default probability for one loan application |
| `POST` | `/explain` | Prediction plus the contribution of each feature (SHAP values) |
| `POST` | `/explain/narrative` | `/explain` plus a plain-language explanation citing EU credit guidelines |

See [Explainability](#explainability) for the last two.

Example request against the live API:

```bash
curl -X POST https://credit-risk-api-uo5r.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "occupation_type": "Laborers"
  }'
```

```json
{"default_probability": 0.0444}
```

Interactive documentation is generated automatically at `/docs`; the root URL
redirects there.

### Design decisions

- **Strict input validation.** Requests are validated with Pydantic before they
  reach the model. Unknown categories, out-of-range values (e.g. a credit score
  above 1, a negative amount) and missing required fields are rejected with a
  `422` response that names the offending field, instead of silently producing
  a meaningless prediction.
- **Optional fields where data can be missing.** External scores, occupation,
  goods price and days employed accept `null`, which the model handles natively
  (e.g. `days_employed: null` for pensioners).
- **Schema checked against the model at startup.** On startup the API compares
  its input schema with the model's `metadata.json` and refuses to start if they
  differ, so a retrained model can never be served with a stale schema.
- **Training/serving consistency.** Each request is rebuilt with the same
  column order and data types used in training before calling the model.

## Explainability

### Feature contributions (`/explain`)

`/explain` returns the contribution of every feature to a prediction as
**SHAP values**, computed with LightGBM's native TreeSHAP (`pred_contrib=True`),
so no extra library is needed on a 512 MB server. Contributions are in log-odds:
the baseline plus all contributions equals the model's output, which a test
verifies.

### Plain-language explanations (`/explain/narrative`)

`/explain/narrative` asks an LLM to turn the SHAP values into a short paragraph
for a loan officer, and relates it to excerpts from the
[EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)](https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring),
returned as numbered `sources` (document, section, page).

Example (abridged):

> The model shows that higher education strongly lowers the risk, and the
> external credit scores of 0.55 and 0.6 also strongly lower the risk [...] the
> loan amount of 500000 and the financed goods price of 450000 each raise the
> risk moderately [...] Overall, the estimated probability of default is 4.4%,
> which is below the baseline of 5.6% for a typical applicant. [...] verify
> borrower information and [...] understand the model's inputs and outputs [1][2].

**Faithfulness is enforced in code, not left to the prompt.** Early versions
invented a currency, called a mid-range score "very good", described 5.5 years of
employment as "short" and claimed the explanation "follows the EBA guidance".
The fixes:

- **Deterministic facts.** The code, not the LLM, interprets the numbers: it
  converts days of employment into years, formats amounts without a currency, and
  labels each contribution as strong, moderate, small or negligible with fixed
  thresholds. The LLM never sees raw column names.
- **Output guardrail.** The response is scanned for value judgements ("very good",
  "high", "relatively"...) and compliance claims ("follows", "complies"...). If any
  are found, the LLM is asked once to rewrite without them, and the event is logged.
- **No decisions.** The prompt forbids recommending approval or rejection: the
  estimate supports a human decision.

**Provider.** Any OpenAI-compatible API works; the defaults point to
[Groq](https://groq.com)'s free tier (`openai/gpt-oss-120b`), which does not use
requests for training. Switching provider or model only means changing the
`LLM_BASE_URL` and `LLM_MODEL` environment variables. Without `LLM_API_KEY` the
endpoint returns `503`; provider failures return `502`.

### Retrieval over regulation (RAG)

The regulatory excerpts come from a retrieval index built offline by
[`build_rag_index.py`](src/credit_risk_api/build_rag_index.py):

1. The guidelines (pp. 12-59 and annexes, pp. 68-76) are split into overlapping
   chunks that keep their section and page, with the repeated page header removed.
2. Chunks are embedded locally with `BAAI/bge-small-en-v1.5` via
   [fastembed](https://github.com/qdrant/fastembed): no API key, no data leaves
   the machine.
3. Each feature maps to a regulatory topic (e.g. the three external scores share
   one), and the most relevant chunk for each topic is **retrieved once, at build
   time**, and reviewed by hand.

Because retrieval queries come from a closed set of topics rather than free
text, the API needs **no embedding model and no vector database at runtime**: it
picks the precomputed excerpts for the three strongest distinct topics of each
application, plus the guidelines' section on automated credit models. For under
200 chunks, NumPy cosine similarity replaces a vector database.

**Retrieval was evaluated, not assumed.** The first index returned passages on
ship financing and commercial real estate for consumer topics. Restricting the
index to the guidelines on loan origination (excluding the executive summary,
consultation feedback and the section on monitoring existing loans), removing
page headers and wording queries for consumer lending fixed most topics. The
evaluation also showed that a similarity threshold would not help: relevant and
irrelevant chunks scored in the same 0.69-0.83 range. Citations in generated
explanations were checked by hand against the cited excerpts.

### Limitations and next steps

- **Guardrails reduce, not eliminate, unwanted phrasing.** A blocklist catches
  known wordings; paraphrases such as "as required by the guidelines" can slip
  through. An LLM-as-a-judge check or human review on a sample would be the next
  step.
- **Citations are faithful but generic.** The explanation cites the excerpts
  correctly but does not always link each one to a specific factor (e.g. that the
  guidelines warn against collateral being a predominant criterion, relevant to
  the goods price).
- **One document.** Income-related retrieval is weak because the guidelines say
  little about consumer income; adding the Consumer Credit Directive
  ((EU) 2023/2225) would cover it.
- **`sources` lists the excerpts given to the LLM**, not only those it cited.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and, for the container,
[Docker](https://www.docker.com/).

```bash
git clone https://github.com/JoaoNunes20/credit-risk-api.git
cd credit-risk-api
uv sync
```

The trained model is included in the repository, so the API, the tests and the
Docker image work right after cloning. To retrain it, download
`application_train.csv` from the
[competition data page](https://www.kaggle.com/competitions/home-credit-default-risk/data)
(you need to accept the competition rules first), place it in `data/raw/` and run
(this overwrites `models/model.joblib` and `models/metadata.json`):

```bash
uv run python -m credit_risk_api.train
```

**Run the API locally:**

```bash
uv run uvicorn credit_risk_api.api:app --reload
```

For `/explain/narrative`, put an API key in a `.env` file (ignored by git) and
load it when starting the server:

```bash
echo 'LLM_API_KEY=your-groq-key' > .env
uv run uvicorn credit_risk_api.api:app --reload --env-file .env
```

**Rebuild the retrieval index** (the index is included in the repository; this
requires the guidelines PDF in `data/regulations/eba_gl_2020_06.pdf`):

```bash
uv run python -m credit_risk_api.build_rag_index
```

**Run the tests** (API and drift monitoring):

```bash
uv run pytest -v
```

**Run the API in Docker:**

```bash
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 credit-risk-api
```

Then open http://127.0.0.1:8000/docs.

## CI/CD

Every push and pull request to `main` runs a [GitHub Actions workflow](.github/workflows/ci.yml)
on a clean machine:

1. **Tests:** installs the exact locked dependencies with uv and runs the
   pytest suite.
2. **Docker build and smoke test:** runs only if the tests pass; builds the
   image, starts a container and checks that `/health` answers.

The service runs on [Render](https://render.com) as a Docker web service, built
directly from the `Dockerfile` in this repository:

- **Continuous deployment gated by CI.** Render deploys a commit to `main` only
  after the CI checks pass, so code with failing tests never reaches
  production; previous versions can be restored with a one-click rollback.
- **Health checks.** Render polls `/health` to verify the service is up.
- **Secrets.** The LLM API key is set as an environment variable in Render and
  in a local `.env` file; it is never committed.
- **Configurable port.** The container listens on the port given by the
  platform's `PORT` environment variable, falling back to 8000 locally, so the
  same image runs unchanged on Render and on a laptop.
- **Model shipped with the code.** At 1.7 MB, the trained model is versioned in
  git alongside the code that serves it. Larger models would instead live in a
  model registry or object storage and be fetched at build or start time.

## Monitoring

### Prediction logging

Every prediction is logged to stdout as one JSON line, which the platform's log
viewer (or any log collector) can search and aggregate:

```json
{"event": "prediction", "default_probability": 0.0444, "latency_ms": 124.38, "inputs": {"amt_credit": 500000.0, "...": "..."}}
```

The log records the predicted probability, the latency and the validated inputs,
which are needed for drift monitoring. The inputs contain no names or
identifiers, but they are personal financial data: in a real lender, logging
them would need a GDPR basis, data minimisation and a retention policy.

### Input drift with the Population Stability Index

[`drift.py`](src/credit_risk_api/drift.py) compares the distribution of the
inputs received by the API with the training distribution, feature by feature,
using the **Population Stability Index (PSI)**, the standard stability metric in
credit scoring:

- At training time, the reference distribution of each feature is saved to
  `models/reference.json`: decile bins for numeric features and frequencies for
  categorical ones. Missing values get their own bin, so a sudden rise in
  missing data (e.g. an upstream source that stops sending a score) is also
  detected.
- Each feature is classified as **stable** (PSI < 0.1), **warning**
  (0.1 to 0.25) or **ALERT** (≥ 0.25).

To test it without real traffic,
[`simulate_traffic.py`](src/credit_risk_api/simulate_traffic.py) generates
2,000 requests from the held-out test set in the API's log format, in two
scenarios. Results:

| Feature | Stable scenario | Shifted scenario (incomes +30%, 30% of cash loans → revolving) |
|---|---|---|
| `AMT_INCOME_TOTAL` | 0.004 (stable) | **0.758 (ALERT)** |
| `NAME_CONTRACT_TYPE` | 0.000 (stable) | **0.478 (ALERT)** |
| Other 9 features | ≤ 0.012 (stable) | unchanged (stable) |

The monitor flags exactly the two features that changed and nothing else. The
non-zero PSI in the stable scenario is sampling noise from a 2,000-request
sample.

Input drift is an **early warning**, not a measure of model quality: whether
the model has actually degraded can only be measured once outcomes are known,
and in credit, defaults surface months after the decision. PSI flags that the
population has changed long before that.

Run it locally (simulation requires the dataset in `data/raw/`):

```bash
uv run python -m credit_risk_api.simulate_traffic
uv run python -m credit_risk_api.drift data/simulated/shifted.jsonl
```

The same command works on prediction logs exported from the platform; any text
before the JSON on each line (such as timestamps) is ignored.

## Project structure

```
credit-risk-api/
├── data/raw/                  # dataset (not tracked)
├── .github/workflows/
│   └── ci.yml                 # CI: tests, Docker build and smoke test
├── models/                    # trained model, metadata and drift reference
│   └── rag/                   # retrieval index: chunks, embeddings, retrieval results
├── notebooks/
│   ├── 01_exploration.ipynb        # exploratory data analysis
│   ├── 02_feature_ablation.ipynb   # impact of removing each feature
│   └── 03_information_value.ipynb  # IV analysis and feature selection
├── src/credit_risk_api/
│   ├── train.py               # training script
│   ├── api.py                 # FastAPI service
│   ├── llm.py                 # LLM explanations: prompt, retrieval context, guardrails
│   ├── build_rag_index.py     # offline retrieval index over the EBA guidelines
│   ├── drift.py               # PSI drift monitoring
│   └── simulate_traffic.py    # simulated traffic for drift checks
├── tests/
│   ├── test_api.py            # API tests
│   └── test_drift.py          # drift monitoring tests
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Roadmap

- [x] **Phase 1 — Data and baseline model:** exploration, feature selection, training script
- [x] **Phase 2 — Prediction API:** FastAPI service with input validation
- [x] **Phase 3 — Containerization and testing:** Docker image and pytest suite
- [x] **Phase 4 — Deployment:** public endpoint on Render with continuous deployment
- [x] **Phase 5 — CI/CD and monitoring:** GitHub Actions, structured prediction logs, PSI drift monitoring
- [x] **Phase 6 — Explainability and LLM layer:** native SHAP values,
  LLM-written explanations with output guardrails, retrieval over EU credit guidelines

## Tech stack

Python 3.13 · uv · pandas · LightGBM · scikit-learn · FastAPI · Pydantic ·
pytest · Docker · GitHub Actions · Render · Groq (OpenAI-compatible API) ·
fastembed
