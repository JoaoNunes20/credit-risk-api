"""Input drift monitoring with the Population Stability Index (PSI).

Compares the distribution of the inputs received by the API with the
distribution of the training data, feature by feature.

Usage, from the project root:
    uv run python -m credit_risk_api.drift path/to/predictions.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = Path("models/reference.json")
EPSILON = 1e-4  # floor for empty bins, avoids log(0)
PSI_WARNING = 0.1
PSI_ALERT = 0.25


def distribution(values: pd.Series, spec: dict) -> np.ndarray:
    """Share of values in each bin of a reference spec. The last bin is 'missing'."""
    missing = values.isna().to_numpy()
    if spec["type"] == "numeric":
        present = values[~missing].astype(float).to_numpy()
        bins = np.searchsorted(spec["edges"], present, side="left")
        counts = np.bincount(bins, minlength=len(spec["edges"]) + 1)
    else:
        present = values[~missing].astype(str)
        counts = np.array([(present == c).sum() for c in spec["categories"]])
    counts = np.append(counts, missing.sum())
    return counts / max(len(values), 1)


def build_reference(X: pd.DataFrame, categorical: list[str]) -> dict:
    """Reference distribution of each feature: deciles for numeric, categories otherwise."""
    reference = {}
    for col in X.columns:
        if col in categorical:
            spec = {"type": "categorical", "categories": [str(c) for c in X[col].cat.categories]}
        else:
            deciles = X[col].dropna().quantile(np.linspace(0.1, 0.9, 9))
            spec = {"type": "numeric", "edges": sorted({float(v) for v in deciles})}
        spec["proportions"] = distribution(X[col], spec).round(6).tolist()
        reference[col] = spec
    return reference


def psi(expected, actual) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    a = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(inputs: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """PSI and status of every feature, most drifted first."""
    rows = []
    for col, spec in reference.items():
        value = psi(spec["proportions"], distribution(inputs[col], spec))
        if value >= PSI_ALERT:
            status = "ALERT"
        elif value >= PSI_WARNING:
            status = "warning"
        else:
            status = "stable"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def read_prediction_logs(path: Path) -> pd.DataFrame:
    """Inputs of all prediction log lines in a file (text before the JSON is ignored)."""
    records = []
    for line in path.read_text().splitlines():
        start = line.find("{")
        if start == -1:
            continue
        try:
            entry = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if entry.get("event") == "prediction":
            records.append(entry["inputs"])
    if not records:
        raise ValueError(f"No prediction log lines found in {path}")
    inputs = pd.DataFrame(records)
    inputs.columns = [c.upper() for c in inputs.columns]
    return inputs


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m credit_risk_api.drift <predictions.jsonl>")
    inputs = read_prediction_logs(Path(sys.argv[1]))
    reference = json.loads(REFERENCE_PATH.read_text())
    print(f"Predictions analysed: {len(inputs)}\n")
    print(drift_report(inputs, reference).to_string(index=False))


if __name__ == "__main__":
    main()
