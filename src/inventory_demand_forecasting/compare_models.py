
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "baseline_metrics.csv"
)

LINEAR_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "linear_regression_metrics.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "model_comparison.csv"
)


REQUIRED_COLUMNS = [
    "model",
    "MAE",
    "RMSE",
    "WAPE_percent",
]


def compare_models(
    baseline_metrics: pd.DataFrame,
    linear_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model metrics and rank models by WAPE."""

    for name, metrics in [
        ("Baseline", baseline_metrics),
        ("Linear Regression", linear_metrics),
    ]:
        missing = set(REQUIRED_COLUMNS) - set(metrics.columns)

        if missing:
            raise ValueError(
                f"{name} metrics are missing columns: {sorted(missing)}"
            )

    combined = pd.concat(
        [
            baseline_metrics[REQUIRED_COLUMNS],
            linear_metrics[REQUIRED_COLUMNS],
        ],
        ignore_index=True,
    )

    for column in ["MAE", "RMSE", "WAPE_percent"]:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="raise",
        )

    if not combined[["MAE", "RMSE", "WAPE_percent"]].notna().all().all():
        raise ValueError("Model metrics contain missing values.")

    if (combined[["MAE", "RMSE", "WAPE_percent"]] < 0).any().any():
        raise ValueError("Model error metrics cannot be negative.")

    return (
        combined.sort_values("WAPE_percent")
        .reset_index(drop=True)
    )


def main() -> None:
    for path in [BASELINE_METRICS_PATH, LINEAR_METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {path}. "
                "Run the baseline and training scripts first."
            )

    baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
    linear_metrics = pd.read_csv(LINEAR_METRICS_PATH)

    comparison = compare_models(
        baseline_metrics,
        linear_metrics,
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
