"""Download the ULB credit card fraud dataset (no account or token needed).

Source: Hugging Face dataset mirror David-Egea/Creditcard-fraud-detection, a copy of the
Kaggle / ULB "creditcard.csv". Kaggle needs a login. OpenML id 1597 has the same rows and
columns (Time, V1..V28, Amount, Class) but declares Time as its row-id attribute, so
loaders such as scikit-learn's fetch_openml drop it from the features, and it is served as
ARFF or parquet rather than the original CSV. The HF mirror is the plain CSV, so one sha256
pins it exactly.

The file is downloaded to a temporary name, checked (sha256, shape, fraud count), then moved
into place in one step, so an interrupted download never leaves a half-written
data/creditcard.csv. An existing file is checked too and never overwritten.

Writes data/creditcard.csv (about 150 MB, ignored by git).
"""
import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
OUT = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path) -> pd.DataFrame:
    sha = sha256_of(path)
    if sha != SHA256:
        raise SystemExit(f"sha256 mismatch for {path}: got {sha}, expected {SHA256}")
    df = pd.read_csv(path)
    assert df.shape == (284807, 31), df.shape
    assert int(df["Class"].sum()) == 492, df["Class"].sum()
    return df


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        print(f"{OUT} exists, checking it")
        df = check(OUT)
    else:
        tmp = OUT.with_suffix(".csv.part")
        print(f"downloading {URL}")
        try:
            urllib.request.urlretrieve(URL, tmp)
            df = check(tmp)
            os.replace(tmp, OUT)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"ok: {OUT} shape={df.shape} frauds={int(df['Class'].sum())} sha256={SHA256}")


if __name__ == "__main__":
    main()
