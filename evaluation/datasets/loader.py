"""
Dataset loading and validation.

DatasetLoader reads evaluation datasets from CSV, JSON, or JSONL files
and validates them against a schema.  It computes a SHA-256 checksum
for dataset lineage tracking (matches the checksum field on the Dataset model).

Supported formats:
  .csv    – pandas read_csv
  .json   – list-of-records or records-under a 'data' key
  .jsonl  – one JSON object per line

Usage::

    loader = DatasetLoader()
    df, checksum = loader.load("/path/to/dataset.csv")
    loader.validate(df, required_columns=["input", "reference"])
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when a dataset fails schema or integrity checks."""


class DatasetLoader:
    """
    Load, checksum, and validate evaluation datasets.

    The loader is stateless; create one per session or share across runs.
    """

    SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl"}

    def load(self, path: str | Path) -> tuple[pd.DataFrame, str]:
        """
        Load a dataset file and return (DataFrame, sha256_hex).

        The checksum is computed over the raw file bytes, so it is
        reproducible and can be stored on the Dataset record.

        Raises:
            FileNotFoundError   – path does not exist
            ValueError          – unsupported file extension or parse error
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        raw_bytes = path.read_bytes()
        checksum = hashlib.sha256(raw_bytes).hexdigest()

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format {ext!r}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        try:
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == ".json":
                data = json.loads(raw_bytes)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and "data" in data:
                    df = pd.DataFrame(data["data"])
                else:
                    df = pd.DataFrame([data])
            elif ext == ".jsonl":
                records = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
                df = pd.DataFrame(records)
            else:
                raise ValueError(f"Extension {ext!r} not handled")
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path.name}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Could not load {path.name}: {exc}") from exc

        return df, checksum

    def validate(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
        min_rows: int = 1,
        max_rows: Optional[int] = None,
    ) -> None:
        """
        Validate a loaded DataFrame.

        Raises:
            DatasetValidationError – on any schema or size violation.
        """
        if len(df) < min_rows:
            raise DatasetValidationError(
                f"Dataset has {len(df)} rows; minimum required is {min_rows}"
            )
        if max_rows is not None and len(df) > max_rows:
            raise DatasetValidationError(
                f"Dataset has {len(df)} rows; maximum allowed is {max_rows}"
            )

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise DatasetValidationError(
                    f"Missing required columns: {missing}. "
                    f"Available: {list(df.columns)}"
                )

    def load_and_validate(
        self,
        path: str | Path,
        required_columns: Optional[list[str]] = None,
        min_rows: int = 1,
        max_rows: Optional[int] = None,
    ) -> tuple[pd.DataFrame, str]:
        """Convenience: load then validate in one call."""
        df, checksum = self.load(path)
        self.validate(df, required_columns=required_columns, min_rows=min_rows, max_rows=max_rows)
        return df, checksum

    @staticmethod
    def checksum_string(text: str) -> str:
        """Return SHA-256 of a UTF-8 string — useful for synthetic/in-memory datasets."""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def checksum_dataframe(df: pd.DataFrame) -> str:
        """Deterministic checksum of a DataFrame (via JSON serialisation)."""
        # sort_keys is not supported by DataFrame.to_json(); sort columns
        # explicitly so the checksum is column-order-independent.
        raw = df[sorted(df.columns)].to_json(orient="records")
        return hashlib.sha256(raw.encode()).hexdigest()
