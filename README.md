# Credit Risk API

[![CI](https://github.com/JoaoNunes20/credit-risk-api/actions/workflows/ci.yml/badge.svg)](https://github.com/JoaoNunes20/credit-risk-api/actions/workflows/ci.yml)

An end-to-end machine learning project that takes a credit default model from
training to a deployed, containerized prediction service.

The focus is on **ML engineering**: serving, validation, testing, packaging,
deployment and monitoring. The model is a deliberately simple baseline; the goal
is to build everything around it the way it would be built in production.

**Live API:** https://credit-risk-api-uo5r.onrender.com (interactive docs)

> Hosted on Render's free tier: after 15 minutes without traffic the service
> sleeps, so the first request can take up to a minute while it wakes up.

> **Status:** work in progress. The model, the prediction API, the test suite,
> the Docker image and the cloud deployment are done; CI/CD and monitoring are
> next. See the [roadmap](#roadmap).

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

A [FastAPI](https://fastapi.tiangolo.com/) service with three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (used by the platform's health checks) |
| `GET` | `/model-info` | Model features, excluded attributes and test AUC |
| `POST` | `/predict` | Default probability for one loan application |

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

**Run the tests:**

```bash
uv run pytest -v
```

**Run the API in Docker:**

```bash
docker build -t credit-risk-api .
docker run --rm -p 8000:8000 credit-risk-api
```

Then open http://127.0.0.1:8000/docs.

## Deployment

The service runs on [Render](https://render.com) as a Docker web service, built
directly from the `Dockerfile` in this repository:

- **Continuous deployment.** Every push to `main` triggers a new build and
  deploy; previous versions can be restored with a one-click rollback.
- **Health checks.** Render polls `/health` to verify the service is up.
- **Configurable port.** The container listens on the port given by the
  platform's `PORT` environment variable, falling back to 8000 locally, so the
  same image runs unchanged on Render and on a laptop.
- **Model shipped with the code.** At 1.7 MB, the trained model is versioned in
  git alongside the code that serves it. Larger models would instead live in a
  model registry or object storage and be fetched at build or start time.

## Project structure

```
credit-risk-api/
├── data/raw/                  # dataset (not tracked)
├── models/                    # trained model and metadata
├── notebooks/
│   ├── 01_exploration.ipynb        # exploratory data analysis
│   ├── 02_feature_ablation.ipynb   # impact of removing each feature
│   └── 03_information_value.ipynb  # IV analysis and feature selection
├── src/credit_risk_api/
│   ├── train.py               # training script
│   └── api.py                 # FastAPI service
├── tests/
│   └── test_api.py            # API test suite
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Roadmap

- [x] **Phase 1 — Data and baseline model:** exploration, feature selection, training script
- [x] **Phase 2 — Prediction API:** FastAPI service with input validation
- [x] **Phase 3 — Containerization and testing:** Docker image and pytest suite
- [x] **Phase 4 — Deployment:** public endpoint on Render with continuous deployment
- [ ] **Phase 5 — CI/CD and monitoring:** GitHub Actions, request logging, drift checks
- [ ] **Phase 6 — LLM explanation layer:** natural-language explanations of
  predictions, combining SHAP values with retrieval over credit regulations

## Tech stack

Python 3.13 · uv · pandas · LightGBM · scikit-learn · FastAPI · Pydantic ·
pytest · Docker · Render
