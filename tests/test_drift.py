"""Unit tests for drift monitoring (synthetic data, no model or dataset needed)."""
import numpy as np
import pandas as pd

from credit_risk_api.drift import build_reference, drift_report, psi, read_prediction_logs


def make_data(n=5000, income_factor=1.0, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=12, sigma=0.5, size=n) * income_factor
    contract = rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1])
    df = pd.DataFrame({"INCOME": income, "CONTRACT": contract})
    df["CONTRACT"] = df["CONTRACT"].astype("category")
    return df


def test_psi_of_identical_distributions_is_zero():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0


def test_similar_population_is_stable():
    reference = build_reference(make_data(seed=0), categorical=["CONTRACT"])
    report = drift_report(make_data(seed=1), reference)
    assert (report["status"] == "stable").all()


def test_income_shift_raises_alert():
    reference = build_reference(make_data(seed=0), categorical=["CONTRACT"])
    report = drift_report(make_data(seed=1, income_factor=1.5), reference).set_index("feature")
    assert report.loc["INCOME", "status"] == "ALERT"
    assert report.loc["CONTRACT", "status"] == "stable"


def test_missing_values_have_their_own_bin():
    data = make_data()
    reference = build_reference(data, categorical=["CONTRACT"])
    with_missing = data.copy()
    with_missing.loc[with_missing.sample(frac=0.5, random_state=0).index, "INCOME"] = np.nan
    report = drift_report(with_missing, reference).set_index("feature")
    assert report.loc["INCOME", "status"] == "ALERT"


def test_read_prediction_logs_ignores_prefixes_and_other_lines(tmp_path):
    log_file = tmp_path / "logs.txt"
    log_file.write_text(
        'INFO: 127.0.0.1 - "GET /health HTTP/1.1" 200 OK\n'
        '2026-11-01 20:45:00 {"event": "prediction", "inputs": {"income": 100.0, "contract": "Cash loans"}}\n'
        '{"event": "prediction", "inputs": {"income": null, "contract": "Revolving loans"}}\n'
    )
    inputs = read_prediction_logs(log_file)
    assert list(inputs.columns) == ["INCOME", "CONTRACT"]
    assert len(inputs) == 2
    assert inputs["INCOME"].isna().sum() == 1
