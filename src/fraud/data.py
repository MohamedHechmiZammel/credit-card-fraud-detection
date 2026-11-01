"""Data loading helpers."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and check it has the expected schema."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Class must be 0 or 1")
