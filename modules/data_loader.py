"""Bounded CSV ingestion; uploaded data never needs to be saved to disk."""
import csv
import io
import numpy as np
import pandas as pd

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200


def load_csv(content: bytes, delimiter: str = ",") -> pd.DataFrame:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose a nonempty CSV file up to 10 MB.")
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text), delimiter=delimiter))
        if not header or any(not name.strip() for name in header):
            raise ValueError("Every column needs a nonempty name.")
        if len(set(header)) != len(header):
            raise ValueError("Column names must be unique.")
        if len(header) > MAX_COLUMNS:
            raise ValueError("Please use at most 200 columns.")
        # Enforce row width instead of allowing pandas to silently infer an index.
        for index, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
            if index > MAX_ROWS:
                raise ValueError("Use a dataset with 1 to 100,000 observations.")
            if row and len(row) != len(header):
                raise ValueError("CSV rows must have the same number of fields as the header.")
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, nrows=MAX_ROWS + 1)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, StopIteration) as exc:
        raise ValueError("Could not read this CSV. Use UTF-8 encoding and check the separator.") from exc
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError("Use a dataset with 1 to 100,000 observations.")
    numeric = frame.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite numeric values are unsupported. Replace them before uploading.")
    return frame


def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 180
    spend = rng.uniform(400, 5000, n).round(2)
    frame = pd.DataFrame({
        "month": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "marketing_spend": spend,
        "sales": (1800 + 2.7 * spend + rng.normal(0, 1200, n)).round(2),
        "satisfaction": rng.integers(1, 6, n),
    })
    frame.loc[[4, 18, 42], "sales"] = np.nan
    return pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
