
"""Streamlit dashboard for the Inventory & Demand Forecasting System."""

from pathlib import Path

import pandas as pd
import streamlit as st

# Find the project root from this source file's location.
# parents[0] = inventory_demand_forecasting
# parents[1] = src
# parents[2] = inventory-demand-forecasting
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store generated evaluation files in the reports directory.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Define the report files that the dashboard will read.
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "xgboost_predictions.csv"
ERROR_SUMMARY_PATH = REPORTS_DIR / "xgboost_error_summary.csv"


def load_csv_report(file_path: Path) -> pd.DataFrame:
    """Load a CSV report and give a clear error if it is unavailable."""

    # Stop loading if the required file has not been generated.
    if not file_path.exists():
        raise FileNotFoundError(f"Required report was not found: {file_path}")

    # Read the CSV file into a pandas DataFrame.
    # A DataFrame allows us to filter, summarize, and visualize the data.
    return pd.read_csv(file_path)


def main() -> None:
    """Render the dashboard's initial page."""

    # Configure the browser tab and use the available page width.
    st.set_page_config(
        page_title="Inventory & Demand Forecasting",
        page_icon="📈",
        layout="wide",
    )

    # Display the main dashboard heading and a short description.
    st.title("Inventory & Demand Forecasting")
    st.write(
        "Explore demand forecasting performance, historical predictions, "
        "and model evaluation results."
    )

    # Load the reports needed by the initial dashboard.
    # Catch missing-file errors so users see a helpful message instead of a traceback.
    try:
        model_comparison = load_csv_report(MODEL_COMPARISON_PATH)
        predictions = load_csv_report(PREDICTIONS_PATH)
        error_summary = load_csv_report(ERROR_SUMMARY_PATH)

    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        # Show an actionable error in the browser.
        st.error(f"Could not load dashboard data: {error}")
        st.info(
            "Generate the forecasting predictions and evaluation reports "
            "before opening the dashboard."
        )

        # Stop rendering data-dependent sections when loading fails.
        st.stop()

    # Display a section for the overall project and dataset information.
    st.header("Dataset and Evaluation Overview")

    # Create three columns so key dataset statistics appear side by side.
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    # Count prediction rows to show how many test observations were evaluated.
    with metric_col1:
        st.metric("Test Predictions", f"{len(predictions):,}")

    # Count distinct stores represented in the prediction dataset.
    with metric_col2:
        if "store" in predictions.columns:
            store_count = predictions["store"].nunique()
            st.metric("Stores Evaluated", f"{store_count:,}")
        else:
            st.metric("Stores Evaluated", "Unavailable")

    # Count distinct items represented in the prediction dataset.
    with metric_col3:
        if "item" in predictions.columns:
            item_count = predictions["item"].nunique()
            st.metric("Items Evaluated", f"{item_count:,}")
        else:
            st.metric("Items Evaluated", "Unavailable")

    # Show the model comparison table so users can inspect model performance.
    st.header("Model Comparison")
    st.caption("Models are compared using the evaluation results generated earlier.")

    # Display the existing comparison data without modifying the source CSV.
    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True,
    )

    # Display the error summary to make the evaluation results easy to inspect.
    st.header("XGBoost Error Summary")

    # Show the summary report as a table for now.
    # We'll convert the main metrics into dashboard cards in the next sub-issue.
    st.dataframe(
        error_summary,
        use_container_width=True,
        hide_index=True,
    )

    # Keep a preview of the prediction data available for debugging and inspection.
    with st.expander("Preview prediction data"):
        st.dataframe(
            predictions.head(20),
            use_container_width=True,
            hide_index=True,
        )

    # Display a footer that identifies the current dashboard milestone.
    st.caption("Phase 08 · Initial dashboard setup")


# Run the application only when Streamlit executes this module directly.
if __name__ == "__main__":
    main()
