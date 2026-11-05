"""
Unit tests for DatasetLoader.

Uses temporary files via tmp_path; no network or database.
"""

from __future__ import annotations

import json
import csv
import io

import pytest

from aieval360.evaluation.datasets.loader import DatasetLoader, DatasetValidationError


@pytest.fixture()
def loader() -> DatasetLoader:
    return DatasetLoader()


@pytest.fixture()
def csv_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("input,reference\nhello,world\nfoo,bar\n")
    return p


@pytest.fixture()
def json_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"input": "hello", "reference": "world"}]))
    return p


@pytest.fixture()
def jsonl_file(tmp_path):
    p = tmp_path / "data.jsonl"
    lines = [json.dumps({"input": "a", "reference": "b"}), json.dumps({"input": "c", "reference": "d"})]
    p.write_text("\n".join(lines))
    return p


class TestDatasetLoaderLoad:
    def test_loads_csv(self, loader, csv_file):
        df, checksum = loader.load(csv_file)
        assert len(df) == 2
        assert "input" in df.columns
        assert len(checksum) == 64  # SHA-256 hex

    def test_loads_json_list(self, loader, json_file):
        df, checksum = loader.load(json_file)
        assert len(df) == 1
        assert "input" in df.columns

    def test_loads_jsonl(self, loader, jsonl_file):
        df, checksum = loader.load(jsonl_file)
        assert len(df) == 2

    def test_loads_json_with_data_key(self, loader, tmp_path):
        p = tmp_path / "wrapped.json"
        p.write_text(json.dumps({"data": [{"input": "x", "reference": "y"}]}))
        df, _ = loader.load(p)
        assert len(df) == 1

    def test_checksum_is_deterministic(self, loader, csv_file):
        _, c1 = loader.load(csv_file)
        _, c2 = loader.load(csv_file)
        assert c1 == c2

    def test_different_files_have_different_checksums(self, loader, tmp_path):
        p1 = tmp_path / "a.csv"
        p2 = tmp_path / "b.csv"
        p1.write_text("input\nhello\n")
        p2.write_text("input\nworld\n")
        _, c1 = loader.load(p1)
        _, c2 = loader.load(p2)
        assert c1 != c2

    def test_missing_file_raises_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.csv")

    def test_unsupported_extension_raises_value_error(self, loader, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"fake parquet")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(p)

    def test_malformed_json_raises_value_error(self, loader, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}")
        with pytest.raises(ValueError):
            loader.load(p)


class TestDatasetLoaderValidate:
    def test_validate_required_columns_pass(self, loader, csv_file):
        df, _ = loader.load(csv_file)
        loader.validate(df, required_columns=["input", "reference"])  # no exception

    def test_validate_required_columns_fail(self, loader, csv_file):
        df, _ = loader.load(csv_file)
        with pytest.raises(DatasetValidationError, match="Missing required columns"):
            loader.validate(df, required_columns=["input", "missing_column"])

    def test_validate_min_rows_pass(self, loader, csv_file):
        df, _ = loader.load(csv_file)
        loader.validate(df, min_rows=2)

    def test_validate_min_rows_fail(self, loader, csv_file):
        df, _ = loader.load(csv_file)
        with pytest.raises(DatasetValidationError):
            loader.validate(df, min_rows=100)

    def test_validate_max_rows_fail(self, loader, csv_file):
        df, _ = loader.load(csv_file)
        with pytest.raises(DatasetValidationError):
            loader.validate(df, max_rows=1)


class TestDatasetLoaderHelpers:
    def test_checksum_string_deterministic(self):
        c1 = DatasetLoader.checksum_string("hello world")
        c2 = DatasetLoader.checksum_string("hello world")
        assert c1 == c2

    def test_checksum_string_different_inputs(self):
        c1 = DatasetLoader.checksum_string("hello")
        c2 = DatasetLoader.checksum_string("world")
        assert c1 != c2

    def test_checksum_dataframe_deterministic(self, csv_file):
        loader = DatasetLoader()
        df, _ = loader.load(csv_file)
        c1 = DatasetLoader.checksum_dataframe(df)
        c2 = DatasetLoader.checksum_dataframe(df)
        assert c1 == c2

    def test_load_and_validate_convenience(self, loader, csv_file):
        df, checksum = loader.load_and_validate(csv_file, required_columns=["input"])
        assert df is not None
        assert checksum is not None
