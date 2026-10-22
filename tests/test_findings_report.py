
"""Tests for the forecast findings report utilities."""

from pathlib import Path

import pandas as pd
import pytest

# Import the functions we want to test from the report generator.
from inventory_demand_forecasting.generate_findings_report import (
    load_report,
    require_columns,
)


def test_load_report_reads_valid_csv(tmp_path: Path) -> None:
    """Confirm that a valid CSV report can be loaded."""

    # Create a temporary CSV file so this test does not modify project reports.
    report_path = tmp_path / "sample_report.csv"

    # Write a small DataFrame to the temporary file.
    expected_data = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "WAPE_percent": [10.35],
        }
    )
    expected_data.to_csv(report_path, index=False)

    # Load the file using the same function as the production report generator.
    actual_data = load_report(report_path)

    # Check that the loaded data matches what we wrote.
    pd.testing.assert_frame_equal(actual_data, expected_data)


def test_load_report_rejects_missing_file(tmp_path: Path) -> None:
    """Confirm that a missing input report produces a clear error."""

    # Create a path without creating the actual file.
    missing_path = tmp_path / "missing_report.csv"

    # The function should not silently accept a missing report.
    with pytest.raises(FileNotFoundError):
        load_report(missing_path)


def test_load_report_rejects_empty_file(tmp_path: Path) -> None:
    """Confirm that an empty report is rejected."""

    # Create an empty file to simulate a report that was not generated correctly.
    empty_path = tmp_path / "empty_report.csv"
    empty_path.write_text("", encoding="utf-8")

    # An empty CSV cannot provide useful forecasting findings.
    with pytest.raises(ValueError):
        load_report(empty_path)


def test_require_columns_accepts_required_columns() -> None:
    """Confirm that a DataFrame with all required columns passes validation."""

    # Create sample data containing both required columns.
    data = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "WAPE_percent": [10.35],
        }
    )

    # Validation should complete without raising an exception.
    require_columns(
        data,
        {"model", "WAPE_percent"},
        "sample report",
    )


def test_require_columns_rejects_missing_columns() -> None:
    """Confirm that missing required columns are detected."""

    # This sample intentionally omits the WAPE_percent column.
    data = pd.DataFrame(
        {
            "model": ["XGBoost"],
        }
    )

    # Missing columns should result in a validation error.
    with pytest.raises(ValueError):
        require_columns(
            data,
            {"model", "WAPE_percent"},
            "sample report",
        )
