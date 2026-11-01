"""Download the ULB credit card fraud dataset (no account or token needed).

Source: Hugging Face dataset mirror David-Egea/Creditcard-fraud-detection, a copy of
the Kaggle / ULB "creditcard.csv". The script checks the shape and the number of
frauds, so a wrong or truncated file fails loudly.
(OpenML id 1597 is also a mirror but it has no Time column, which the original
project used, so it is not used here.)

Writes data/creditcard.csv (about 150 MB, ignored by git).
"""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/David-Egea/Creditcard-fraud-detection/resolve/main/creditcard.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    if not OUT.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, OUT)
    df = pd.read_csv(OUT)
    assert df.shape == (284807, 31), df.shape
    assert int(df["Class"].sum()) == 492, df["Class"].sum()
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"ok: {OUT} shape={df.shape} frauds={int(df['Class'].sum())} sha256={sha}")


if __name__ == "__main__":
    main()
