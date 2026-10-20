
"""Compare baseline, Linear Regression, and XGBoost models."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.csv"
LINEAR_METRICS_PATH = REPORTS_DIR / "linear_regression_metrics.csv"
XGBOOST_METRICS_PATH = REPORTS_DIR / "xgboost_metrics.csv"

OUTPUT_PATH = REPORTS_DIR / "model_comparison.csv"

REQUIRED_COLUMNS = [
    "model",
    "MAE",
    "RMSE",
    "WAPE_percent",
]


def compare_models(*metric_tables: pd.DataFrame) -> pd.DataFrame:
    """Combine model metrics and rank models by WAPE."""

    if not metric_tables:
        raise ValueError("At least one metrics table is required.")

    validated_tables = []

    for index, metrics in enumerate(metric_tables, start=1):
        missing = set(REQUIRED_COLUMNS) - set(metrics.columns)

        if missing:
            raise ValueError(
                f"Metrics table {index} is missing columns: "
                f"{sorted(missing)}"
            )

        selected = metrics[REQUIRED_COLUMNS].copy()

        for column in ["MAE", "RMSE", "WAPE_percent"]:
            selected[column] = pd.to_numeric(
                selected[column],
                errors="raise",
            )

        if not selected[REQUIRED_COLUMNS].notna().all().all():
            raise ValueError(
                f"Metrics table {index} contains missing values."
            )

        if (selected[["MAE", "RMSE", "WAPE_percent"]] < 0).any().any():
            raise ValueError(
                f"Metrics table {index} contains negative error metrics."
            )

        validated_tables.append(selected)

    combined = pd.concat(validated_tables, ignore_index=True)

    if combined["model"].duplicated().any():
        duplicates = combined.loc[
            combined["model"].duplicated(keep=False), "model"
        ].tolist()

        raise ValueError(
            f"Duplicate model names found: {duplicates}"
        )

    return (
        combined.sort_values("WAPE_percent")
        .reset_index(drop=True)
    )


def main() -> None:
    """Load metrics, create a ranking, and save the report."""

    metric_paths = [
        BASELINE_METRICS_PATH,
        LINEAR_METRICS_PATH,
        XGBOOST_METRICS_PATH,
    ]

    for path in metric_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Metrics file not found: {path}. "
                "Run all baseline and model training scripts first."
            )

    baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
    linear_metrics = pd.read_csv(LINEAR_METRICS_PATH)
    xgboost_metrics = pd.read_csv(XGBOOST_METRICS_PATH)

    comparison = compare_models(
        baseline_metrics,
        linear_metrics,
        xgboost_metrics,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print("\n=== MODEL COMPARISON — LOWER IS BETTER ===")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nComparison report saved to: {OUTPUT_PATH}")
    print(f"Best model by WAPE: {comparison.iloc[0]['model']}")


if __name__ == "__main__":
    main()
