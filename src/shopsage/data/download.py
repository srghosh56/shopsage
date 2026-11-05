"""Download UCI Online Retail II and convert both Excel sheets to one raw Parquet file.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (licence CC BY 4.0).
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from shopsage.config import RAW_DIR, UCI_URL

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]
EXPECTED_SHEETS = {"Year 2009-2010", "Year 2010-2011"}
RAW_PARQUET = RAW_DIR / "online_retail_ii.parquet"


def check_raw(df: pd.DataFrame) -> None:
    """Sanity checks on the raw download; raise if the file is not what we expect."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(df) < 1_000_000:
        raise ValueError(f"expected >1M rows, got {len(df):,}")
    lo, hi = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    if lo.year != 2009 or hi.year != 2011:
        raise ValueError(f"unexpected date range {lo} .. {hi}")


def download(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PARQUET.exists() and not force:
        log.info("raw parquet exists: %s", RAW_PARQUET)
        return pd.read_parquet(RAW_PARQUET)

    log.info("downloading %s", UCI_URL)
    resp = requests.get(UCI_URL, timeout=300)
    resp.raise_for_status()
    zip_path = RAW_DIR / "online_retail_ii.zip"
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        sheets = pd.read_excel(
            io.BytesIO(zf.read(xlsx_name)),
            sheet_name=None,
            dtype={"Invoice": str, "StockCode": str, "Description": str},
        )
    if set(sheets) != EXPECTED_SHEETS:
        raise ValueError(f"unexpected sheets: {list(sheets)}")
    df = pd.concat(sheets.values(), ignore_index=True)
    check_raw(df)
    df.to_parquet(RAW_PARQUET, index=False)
    log.info("wrote %s (%s rows)", RAW_PARQUET, f"{len(df):,}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download()
