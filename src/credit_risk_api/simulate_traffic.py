"""Simulate API traffic as prediction log lines, to exercise drift monitoring.

Writes two files to data/simulated/ (not tracked by git):
- stable.jsonl:  applications from the held-out test set, as the API would log them
- shifted.jsonl: the same applications with incomes 30% higher and 30% of
                 cash loans turned into revolving loans

Run from the project root (requires data/raw/application_train.csv):
    uv run python -m credit_risk_api.simulate_traffic
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_api.train import FEATURES, TARGET, load_data

OUTPUT_DIR = Path("data/simulated")
N_REQUESTS = 2000


def write_logs(X: pd.DataFrame, path: Path) -> None:
    with path.open("w") as f:
        for row in X.to_dict(orient="records"):
            inputs = {k.lower(): (None if pd.isna(v) else v) for k, v in row.items()}
            f.write(json.dumps({"event": "prediction", "inputs": inputs}) + "\n")


def main() -> None:
    df = load_data()
    # Same split as training, so the simulated traffic never overlaps the training data.
    _, X_test, _, _ = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )
    stable = X_test.sample(N_REQUESTS, random_state=0)

    shifted = stable.copy()
    shifted["AMT_INCOME_TOTAL"] = shifted["AMT_INCOME_TOTAL"] * 1.3
    rng = np.random.default_rng(0)
    to_revolving = (shifted["NAME_CONTRACT_TYPE"] == "Cash loans") & (rng.random(len(shifted)) < 0.3)
    shifted.loc[to_revolving, "NAME_CONTRACT_TYPE"] = "Revolving loans"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_logs(stable, OUTPUT_DIR / "stable.jsonl")
    write_logs(shifted, OUTPUT_DIR / "shifted.jsonl")
    print(f"Wrote {N_REQUESTS} simulated requests to {OUTPUT_DIR}/stable.jsonl and shifted.jsonl")


if __name__ == "__main__":
    main()
