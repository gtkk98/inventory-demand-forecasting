
import pandas as pd

from inventory_demand_forecasting.validation import validate_sales_data


def make_valid_data() -> pd.DataFrame:
    """Return a small, valid sales dataset for testing."""
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "store": [1, 1],
            "item": [10, 10],
            "sales": [12, 15],
        }
    )


def test_valid_data_passes_basic_checks():
    df = make_valid_data()

    report = validate_sales_data(df)

    assert (
        report.loc[
            report["check"] == "Required columns", "status"
        ].iloc[0]
        == "PASS"
    )

    assert (
        report.loc[
            report["check"] == "Unique store-product-date keys",
            "issue_count",
        ].iloc[0]
        == 0
    )


def test_duplicate_store_item_date_is_detected():
    df = make_valid_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = validate_sales_data(df)

    duplicate_check = report[
        report["check"] == "Unique store-product-date keys"
    ].iloc[0]

    assert duplicate_check["status"] == "FAIL"
    assert duplicate_check["issue_count"] == 2


def test_invalid_date_is_detected():
    df = make_valid_data()
    df.loc[0, "date"] = "not-a-real-date"

    report = validate_sales_data(df)

    date_check = report[report["check"] == "Invalid dates"].iloc[0]

    assert date_check["status"] == "FAIL"
    assert date_check["issue_count"] == 1


def test_negative_sales_is_detected():
    df = make_valid_data()
    df.loc[0, "sales"] = -5

    report = validate_sales_data(df)

    sales_check = report[
        report["check"] == "Non-negative sales"
    ].iloc[0]

    assert sales_check["status"] == "FAIL"
    assert sales_check["issue_count"] == 1
