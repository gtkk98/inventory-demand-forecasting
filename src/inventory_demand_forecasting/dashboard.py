
"""Streamlit dashboard for the Inventory & Demand Forecasting System.

This dashboard displays model performance, daily sales forecasts, forecast
errors, and store/item-level predictions. It also supports CSV downloads.
"""

# Import Path to construct file paths that work on Windows and other systems.
from pathlib import Path

# Import NumPy to detect infinite and other non-finite numeric values.
import numpy as np

# Import pandas for loading, validating, filtering, and aggregating CSV data.
import pandas as pd

# Import Streamlit to create the interactive web dashboard.
import streamlit as st

# Import date for the forecast-date selector.
from datetime import date

# Import HTTP errors so the dashboard can handle API failures.
import httpx

# Import the functions that communicate with the forecasting API.
from inventory_demand_forecasting.api_client import (
    check_api_health,
    request_forecast,
)

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS AND DASHBOARD CONFIGURATION
# ---------------------------------------------------------------------------

# Resolve the project root from this file's location:
# project_root/src/inventory_demand_forecasting/dashboard.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the directory containing generated reports.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Define the CSV containing model comparison metrics.
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# Define the CSV containing XGBoost's actual and predicted sales.
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_predictions.csv"
)

# Define the report containing overall XGBoost error statistics.
ERROR_SUMMARY_PATH = REPORTS_DIR / "xgboost_error_summary.csv"

# Configure the browser tab and use the available page width.
st.set_page_config(
    page_title="Inventory & Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 2. DATA LOADING AND VALIDATION HELPERS
# ---------------------------------------------------------------------------

def load_csv_report(file_path: Path) -> pd.DataFrame:
    """Load a CSV report and reject missing, empty, or unreadable files."""

    # Make sure the requested report actually exists.
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required CSV file was not found: {file_path}"
        )

    # Load the CSV into a pandas DataFrame.
    data = pd.read_csv(file_path)

    # Reject reports with no rows or no columns.
    if data.empty or len(data.columns) == 0:
        raise ValueError(f"The CSV report is empty: {file_path}")

    # Return the loaded data to the calling function.
    return data


def validate_model_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Validate model metrics and rank models by lowest WAPE."""

    # Define the columns needed to compare forecasting models.
    required_columns = {
        "model",
        "MAE",
        "RMSE",
        "WAPE_percent",
    }

    # Find any required columns that the supplied report does not contain.
    missing_columns = required_columns.difference(data.columns)

    # Reject incomplete model comparison reports.
    if missing_columns:
        raise ValueError(
            "Model comparison is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Work on a copy so the original report is not modified.
    ranked_data = data.copy()

    # Require every model to have a non-empty name.
    if ranked_data["model"].isna().any():
        raise ValueError("Model names cannot contain missing values.")

    # Convert model names to strings and remove surrounding whitespace.
    ranked_data["model"] = ranked_data["model"].astype(str).str.strip()

    # Reject empty model names.
    if ranked_data["model"].eq("").any():
        raise ValueError("Model names cannot be empty.")

    # Define the numeric columns used to evaluate model performance.
    metric_columns = ["MAE", "RMSE", "WAPE_percent"]

    # Convert metric columns to numeric values; reject unparseable values.
    for metric in metric_columns:
        ranked_data[metric] = pd.to_numeric(
            ranked_data[metric],
            errors="raise",
        )

    # Convert the metrics into a NumPy array for validation.
    metric_values = ranked_data[metric_columns].to_numpy(dtype=float)

    # Reject NaN and positive/negative infinity before ranking models.
    if not np.isfinite(metric_values).all():
        raise ValueError(
            "Model comparison metrics contain non-finite values."
        )

    # MAE, RMSE, and WAPE cannot be negative.
    if (ranked_data[metric_columns] < 0).any().any():
        raise ValueError(
            "Model comparison metrics cannot contain negative values."
        )

    # At least one model must be available for comparison.
    if ranked_data.empty:
        raise ValueError("No models are available for comparison.")

    # Rank the models from lowest to highest WAPE.
    ranked_data = ranked_data.sort_values(
        by="WAPE_percent",
        ascending=True,
    ).reset_index(drop=True)

    # Return the validated model report.
    return ranked_data


def prepare_prediction_chart(data: pd.DataFrame) -> pd.DataFrame:
    """Validate predictions and aggregate actual/predicted sales by date."""

    # Specify the fields needed to build the daily sales chart.
    required_columns = {
        "date",
        "actual_sales",
        "predicted_sales",
    }

    # Check that the prediction data contains all required fields.
    missing_columns = required_columns.difference(data.columns)

    # Reject a prediction file with missing chart fields.
    if missing_columns:
        raise ValueError(
            "Prediction data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Work on a copy to protect the caller's DataFrame.
    chart_data = data.copy()

    # Parse the dates; invalid date values become NaT.
    chart_data["date"] = pd.to_datetime(
        chart_data["date"],
        errors="coerce",
    )

    # Convert actual and predicted sales into numeric values.
    for column in ["actual_sales", "predicted_sales"]:
        chart_data[column] = pd.to_numeric(
            chart_data[column],
            errors="coerce",
        )

    # Remove rows with invalid dates or missing/non-numeric sales.
    chart_data = chart_data.dropna(
        subset=["date", "actual_sales", "predicted_sales"]
    )

    # Reject infinite sales values, which cannot produce reliable charts.
    numeric_values = chart_data[
        ["actual_sales", "predicted_sales"]
    ].to_numpy(dtype=float)

    # Keep only rows where both sales values are finite.
    finite_rows = np.isfinite(numeric_values).all(axis=1)
    chart_data = chart_data.loc[finite_rows].copy()

    # Stop if no usable prediction records remain.
    if chart_data.empty:
        raise ValueError(
            "No valid prediction records are available for charting."
        )

    # Aggregate all store-item observations into daily sales totals.
    daily_data = (
        chart_data.groupby("date", as_index=False)[
            ["actual_sales", "predicted_sales"]
        ]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Return one row per date for the daily sales chart.
    return daily_data


def prepare_daily_error_chart(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate signed and absolute errors for each prediction row."""

    # Verify that dates and both sales values are present.
    required_columns = {
        "date",
        "actual_sales",
        "predicted_sales",
    }

    # Identify any required fields that are absent.
    missing_columns = required_columns.difference(data.columns)

    # Reject data that cannot be used to calculate errors.
    if missing_columns:
        raise ValueError(
            "Prediction data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Copy the input so error calculations do not modify the source data.
    error_data = data.copy()

    # Convert dates into pandas datetime values.
    error_data["date"] = pd.to_datetime(
        error_data["date"],
        errors="coerce",
    )

    # Ensure actual and predicted sales are numeric.
    for column in ["actual_sales", "predicted_sales"]:
        error_data[column] = pd.to_numeric(
            error_data[column],
            errors="coerce",
        )

    # Remove rows that cannot be used in the error calculations.
    error_data = error_data.dropna(
        subset=["date", "actual_sales", "predicted_sales"]
    )

    # Remove rows containing infinite actual or predicted values.
    numeric_values = error_data[
        ["actual_sales", "predicted_sales"]
    ].to_numpy(dtype=float)

    # Keep only rows where both numeric values are finite.
    finite_rows = np.isfinite(numeric_values).all(axis=1)
    error_data = error_data.loc[finite_rows].copy()

    # Calculate signed error: positive means underprediction.
    error_data["error"] = (
        error_data["actual_sales"] - error_data["predicted_sales"]
    )

    # Calculate absolute error without the direction of the error.
    error_data["absolute_error"] = error_data["error"].abs()

    # Return one row per valid prediction, with both error columns.
    return error_data.reset_index(drop=True)


def filter_predictions(
    data: pd.DataFrame,
    selected_store,
    selected_item,
) -> pd.DataFrame:
    """Filter predictions by store and item; None means no restriction."""

    # Require the identifiers needed for store/item filtering.
    required_columns = {"store", "item"}
    missing_columns = required_columns.difference(data.columns)

    # Reject data that cannot support the requested filters.
    if missing_columns:
        raise ValueError(
            "Prediction data is missing filter columns: "
            f"{sorted(missing_columns)}"
        )

    # Begin with all rows selected.
    filtered_data = data.copy()

    # Apply the store filter only when a particular store was selected.
    if selected_store is not None:
        filtered_data = filtered_data[
            filtered_data["store"] == selected_store
        ]

    # Apply the item filter only when a particular item was selected.
    if selected_item is not None:
        filtered_data = filtered_data[
            filtered_data["item"] == selected_item
        ]

    # Reset row numbers so the filtered table is easier to inspect.
    return filtered_data.reset_index(drop=True)


def calculate_sales_metrics(data: pd.DataFrame) -> dict:
    """Calculate MAE, RMSE, and WAPE for a prediction DataFrame."""

    # Return zero-valued metrics when no records are available.
    if data.empty:
        return {"MAE": 0.0, "RMSE": 0.0, "WAPE_percent": 0.0}

    # Extract the actual and predicted sales as numeric arrays.
    actual = pd.to_numeric(
        data["actual_sales"],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        data["predicted_sales"],
        errors="coerce",
    ).to_numpy(dtype=float)

    # Keep only pairs containing finite numeric values.
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    # Return zero-valued metrics if no valid pairs remain.
    if len(actual) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "WAPE_percent": 0.0}

    # Calculate the signed errors for all valid observations.
    errors = actual - predicted

    # Calculate mean absolute error in sales units.
    mae = float(np.mean(np.abs(errors)))

    # Calculate root mean squared error in sales units.
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    # Calculate WAPE relative to total actual sales.
    actual_total = float(np.sum(np.abs(actual)))
    wape = (
        float(np.sum(np.abs(errors)) / actual_total * 100)
        if actual_total > 0
        else 0.0
    )

    # Return the three metrics in a consistent dictionary.
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


def make_csv_download(data: pd.DataFrame) -> bytes:
    """Convert a DataFrame into UTF-8 CSV bytes for Streamlit downloads."""

    # Exclude the DataFrame's row index from the exported file.
    csv_text = data.to_csv(index=False)

    # Encode the CSV as bytes for the download button.
    return csv_text.encode("utf-8")


# ---------------------------------------------------------------------------
# 3. DASHBOARD PRESENTATION HELPERS
# ---------------------------------------------------------------------------

def show_metric_explanations() -> None:
    """Display explanations of the model evaluation metrics."""

    # Put detailed definitions in an expandable section.
    with st.expander("Understand the forecasting metrics"):

        # Explain average absolute prediction error.
        st.markdown(
            """
            **MAE — Mean Absolute Error**

            The average absolute difference between actual and predicted
            sales. An MAE of 6.09 means the average absolute error is
            approximately 6.09 sales units per evaluated observation.
            """
        )

        # Explain how RMSE treats large prediction errors.
        st.markdown(
            """
            **RMSE — Root Mean Squared Error**

            Measures prediction error in sales units and gives relatively
            more weight to large mistakes. Lower values are generally better.
            """
        )

        # Explain relative error using WAPE.
        st.markdown(
            """
            **WAPE — Weighted Absolute Percentage Error**

            Total absolute prediction error divided by total actual sales,
            expressed as a percentage. A WAPE of 10.35% means the sum of
            absolute errors is about 10.35% of total actual sales.
            """
        )

        # Remind users to compare models over equivalent evaluation periods.
        st.caption(
            "Compare models on the same dates and observations. Also inspect "
            "performance by weekday, month, store, and item."
        )


def show_metric_cards(metrics: dict) -> None:
    """Display MAE, RMSE, and WAPE in three metric cards."""

    # Create three equally sized columns for the evaluation metrics.
    mae_column, rmse_column, wape_column = st.columns(3)

    # Display average absolute error.
    mae_column.metric(
        "MAE",
        f"{metrics['MAE']:.4f}",
        help="Mean absolute error, measured in sales units.",
    )

    # Display root mean squared error.
    rmse_column.metric(
        "RMSE",
        f"{metrics['RMSE']:.4f}",
        help="Root mean squared error, measured in sales units.",
    )

    # Display weighted absolute percentage error.
    wape_column.metric(
        "WAPE",
        f"{metrics['WAPE_percent']:.4f}%",
        help="Total absolute error divided by total actual sales.",
    )


# ---------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# ---------------------------------------------------------------------------

def main() -> None:
    """Load project reports and render the interactive dashboard."""

    # Display the dashboard title and project description.
    st.title("📈 Inventory & Demand Forecasting System")
    st.caption(
        "Explore demand forecasts, compare models, and investigate "
        "forecast errors across dates, stores, and items."
    )

    # Load and validate each report before building the dashboard.
    try:
        model_comparison = validate_model_comparison(
            load_csv_report(MODEL_COMPARISON_PATH)
        )

        # Load the row-level XGBoost predictions.
        predictions = load_csv_report(PREDICTIONS_PATH)

        # Verify that the prediction file has all required fields.
        prediction_columns = {
            "date",
            "store",
            "item",
            "actual_sales",
            "predicted_sales",
        }

        # Identify required prediction columns that are absent.
        missing_prediction_columns = prediction_columns.difference(
            predictions.columns
        )

        # Stop early if the prediction file is missing important fields.
        if missing_prediction_columns:
            raise ValueError(
                "Prediction file is missing required columns: "
                f"{sorted(missing_prediction_columns)}"
            )

        # Parse dates so date filters and charts can work reliably.
        predictions["date"] = pd.to_datetime(
            predictions["date"],
            errors="coerce",
        )

        # Convert the sales columns into numeric values.
        for column in ["actual_sales", "predicted_sales"]:
            predictions[column] = pd.to_numeric(
                predictions[column],
                errors="coerce",
            )

        # Remove records with invalid dates or missing sales values.
        predictions = predictions.dropna(
            subset=["date", "actual_sales", "predicted_sales"]
        ).copy()

        # Reject infinite sales values before displaying results.
        prediction_values = predictions[
            ["actual_sales", "predicted_sales"]
        ].to_numpy(dtype=float)

        # Keep only rows with finite actual and predicted sales.
        finite_prediction_rows = np.isfinite(prediction_values).all(axis=1)
        predictions = predictions.loc[finite_prediction_rows].copy()

        # Reject an empty prediction dataset.
        if predictions.empty:
            raise ValueError(
                "No valid prediction records remain after validation."
            )

        # Load the overall error summary when it is available.
        if ERROR_SUMMARY_PATH.is_file():
            error_summary = load_csv_report(ERROR_SUMMARY_PATH)
        else:
            error_summary = pd.DataFrame()

    # Present a useful error message if the dashboard cannot load its data.
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.info(
            "Check that the required CSV reports exist and contain "
            "the expected columns. Generate the model reports first if needed."
        )
        return

    # Prepare daily totals for the overview and daily sales chart.
    try:
        daily_sales = prepare_prediction_chart(predictions)

    # Stop the dashboard cleanly if daily aggregation fails.
    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Could not prepare daily sales data: {error}")
        return

    # Get the best model using the lowest WAPE in the validated report.
    best_model = model_comparison.iloc[0]

    # Calculate XGBoost's metrics directly from the loaded predictions.
    xgboost_metrics = calculate_sales_metrics(predictions)

    # Display the headline results before the navigation tabs.
    st.subheader("Forecasting Overview")

    # Show the number of evaluated observations.
    observation_column, model_column, date_column = st.columns(3)

    observation_column.metric(
        "Valid prediction observations",
        f"{len(predictions):,}",
    )

    # Show the best model name based on the lowest reported WAPE.
    model_column.metric(
        "Best model by WAPE",
        str(best_model["model"]),
    )

    # Display the period covered by the prediction dataset.
    date_column.metric(
        "Evaluation period",
        f"{predictions['date'].min():%Y-%m-%d} to "
        f"{predictions['date'].max():%Y-%m-%d}",
    )

    # Create five tabs, including the new live forecasting interface.
    (
        overview_tab,
        sales_tab,
        errors_tab,
        explorer_tab,
        live_forecast_tab,
    ) = st.tabs(
        [
            "Overview & Models",
            "Daily Sales",
            "Error Analysis",
            "Store & Item Explorer",
            "Live Forecast",
        ]
    )

    # -----------------------------------------------------------------------
    # TAB 1: OVERVIEW AND MODEL COMPARISON
    # -----------------------------------------------------------------------

    with overview_tab:
        # Explain that the model metrics are based on the evaluated data.
        st.subheader("XGBoost Evaluation Metrics")

        # Display the MAE, RMSE, and WAPE cards.
        show_metric_cards(xgboost_metrics)

        # Explain the meaning and limitations of each metric.
        show_metric_explanations()

        # Display the model comparison table.
        st.subheader("Model Comparison")

        # Show the models ordered by lowest WAPE.
        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True,
        )

        # Allow users to export all model comparison results.
        st.download_button(
            label="Download model comparison (CSV)",
            data=make_csv_download(model_comparison),
            file_name="model_comparison.csv",
            mime="text/csv",
            help="Export the evaluation metrics for all compared models.",
            key="download_model_comparison",
        )

        # Build a bar chart to compare WAPE across the models.
        st.subheader("WAPE by Model")

        # Use model names as the chart index and WAPE as the plotted value.
        wape_chart = model_comparison.set_index("model")[
            ["WAPE_percent"]
        ]

        # Display the comparison chart.
        st.bar_chart(wape_chart)

        # Display the overall error summary when the report is available.
        if not error_summary.empty:
            st.subheader("Saved XGBoost Error Summary")

            # Show the existing summary without assuming a fixed row schema.
            st.dataframe(
                error_summary,
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------------------------
    # TAB 2: DAILY SALES AND DATE FILTERING
    # -----------------------------------------------------------------------

    with sales_tab:
        # Introduce the actual-versus-predicted sales chart.
        st.subheader("Actual vs Predicted Daily Sales")

        # Derive the valid date limits from the available daily data.
        min_date = daily_sales["date"].min().date()
        max_date = daily_sales["date"].max().date()

        # Let users choose the time period they want to inspect.
        selected_dates = st.date_input(
            "Select evaluation date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="sales_date_range",
        )

        # A range selector may return a single date while the user is
        # selecting the second date. Handle that state without crashing.
        if not isinstance(selected_dates, (tuple, list)) or len(
            selected_dates
        ) != 2:
            st.info(
                "Select both a start date and an end date to view daily sales."
            )

        else:
            # Unpack the selected start and end dates.
            start_date, end_date = selected_dates

            # Guard against an invalid date range.
            if start_date > end_date:
                st.warning(
                    "The start date must be on or before the end date."
                )

            else:
                # Filter daily totals using the selected date range.
                filtered_daily_sales = daily_sales[
                    (
                        daily_sales["date"].dt.date >= start_date
                    )
                    & (
                        daily_sales["date"].dt.date <= end_date
                    )
                ].copy()

                # Display a message rather than rendering an empty chart.
                if filtered_daily_sales.empty:
                    st.info(
                        "No sales records match this date range. "
                        "Please choose a different range."
                    )

                else:
                    # Display the filtered actual and predicted daily totals.
                    sales_chart = filtered_daily_sales.set_index("date")[
                        ["actual_sales", "predicted_sales"]
                    ]

                    # Draw both sales series on the same chart.
                    st.line_chart(sales_chart)

                    # Calculate total actual and predicted sales for the range.
                    actual_total = float(
                        filtered_daily_sales["actual_sales"].sum()
                    )
                    predicted_total = float(
                        filtered_daily_sales["predicted_sales"].sum()
                    )

                    # Calculate the difference in total sales.
                    sales_difference = predicted_total - actual_total

                    # Display the totals in three columns.
                    actual_column, predicted_column, difference_column = (
                        st.columns(3)
                    )

                    actual_column.metric(
                        "Actual sales total",
                        f"{actual_total:,.0f}",
                    )

                    predicted_column.metric(
                        "Predicted sales total",
                        f"{predicted_total:,.0f}",
                    )

                    difference_column.metric(
                        "Predicted minus actual",
                        f"{sales_difference:,.2f}",
                    )

                    # Explain the direction of the total difference.
                    if sales_difference < 0:
                        st.caption(
                            "The predicted total is lower than the actual "
                            "total for the selected period."
                        )
                    elif sales_difference > 0:
                        st.caption(
                            "The predicted total is higher than the actual "
                            "total for the selected period."
                        )
                    else:
                        st.caption(
                            "The predicted and actual totals are equal "
                            "for the selected period."
                        )

    # -----------------------------------------------------------------------
    # TAB 3: FORECAST ERROR ANALYSIS
    # -----------------------------------------------------------------------

    with errors_tab:
        # Explain how signed and absolute errors should be interpreted.
        st.subheader("Daily Forecast Error Analysis")

        st.caption(
            "Signed error = actual sales − predicted sales. Positive values "
            "indicate underprediction; negative values indicate overprediction."
        )

        # Calculate errors for each valid store-item prediction.
        try:
            error_data = prepare_daily_error_chart(predictions)

        # Handle unexpected invalid error data without crashing the app.
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Could not calculate forecast errors: {error}")
            error_data = pd.DataFrame()

        # Continue only if error calculations produced valid records.
        if error_data.empty:
            st.info("No valid prediction errors are available to display.")

        else:
            # Aggregate signed errors and absolute errors by date.
            daily_errors = (
                error_data.groupby("date", as_index=False)[
                    ["error", "absolute_error"]
                ]
                .sum()
                .sort_values("date")
                .reset_index(drop=True)
            )

            # Use the same date range as the Daily Sales tab.
            error_date_range = st.date_input(
                "Select error analysis date range",
                value=(
                    daily_errors["date"].min().date(),
                    daily_errors["date"].max().date(),
                ),
                min_value=daily_errors["date"].min().date(),
                max_value=daily_errors["date"].max().date(),
                key="error_date_range",
            )

            # Handle incomplete date selection.
            if not isinstance(error_date_range, (tuple, list)) or len(
                error_date_range
            ) != 2:
                st.info(
                    "Select both dates to view forecast errors."
                )

            else:
                # Read the selected error analysis date range.
                error_start, error_end = error_date_range

                # Reject an invalid range before filtering.
                if error_start > error_end:
                    st.warning(
                        "The start date must be on or before the end date."
                    )

                else:
                    # Keep daily errors within the selected date range.
                    filtered_errors = daily_errors[
                        (
                            daily_errors["date"].dt.date >= error_start
                        )
                        & (
                            daily_errors["date"].dt.date <= error_end
                        )
                    ].copy()

                    # Avoid rendering a chart with no error observations.
                    if filtered_errors.empty:
                        st.info(
                            "No forecast errors match this date range."
                        )

                    else:
                        # Display the daily signed error series.
                        st.markdown("**Daily signed error**")

                        st.line_chart(
                            filtered_errors.set_index("date")[["error"]]
                        )

                        # Display the daily absolute error series.
                        st.markdown("**Daily absolute error**")

                        st.line_chart(
                            filtered_errors.set_index("date")[
                                ["absolute_error"]
                            ]
                        )

                        # Summarize the direction and magnitude of errors.
                        total_signed_error = float(
                            filtered_errors["error"].sum()
                        )
                        total_absolute_error = float(
                            filtered_errors["absolute_error"].sum()
                        )

                        signed_column, absolute_column = st.columns(2)

                        signed_column.metric(
                            "Sum of signed errors",
                            f"{total_signed_error:,.2f}",
                        )

                        absolute_column.metric(
                            "Sum of absolute errors",
                            f"{total_absolute_error:,.2f}",
                        )

        # Keep the saved error report available for inspection.
        if not error_summary.empty:
            with st.expander("View saved overall error summary"):
                st.dataframe(
                    error_summary,
                    use_container_width=True,
                    hide_index=True,
                )

    # -----------------------------------------------------------------------
    # TAB 4: STORE AND ITEM EXPLORER
    # -----------------------------------------------------------------------

    with explorer_tab:
        # Introduce the store/item investigation section.
        st.subheader("Explore Predictions by Store and Item")

        # Explain how the selectors affect the displayed observations.
        st.caption(
            "Select a store and item to inspect their actual sales, "
            "predictions, and evaluation metrics."
        )

        # Sort the available store identifiers for consistent selection.
        available_stores = sorted(
            predictions["store"].dropna().unique().tolist()
        )

        # Sort the available item identifiers for consistent selection.
        available_items = sorted(
            predictions["item"].dropna().unique().tolist()
        )

        # Stop if the data has no usable store or item identifiers.
        if not available_stores or not available_items:
            st.warning(
                "The prediction data contains no valid store or item IDs."
            )

        else:
            # Place the store and item selectors side by side.
            store_column, item_column = st.columns(2)

            # Let the user choose a store.
            selected_store = store_column.selectbox(
                "Select store",
                options=available_stores,
                key="selected_store",
            )

            # Let the user choose an item.
            selected_item = item_column.selectbox(
                "Select item",
                options=available_items,
                key="selected_item",
            )

            # Apply the chosen store and item filters.
            filtered_predictions = filter_predictions(
                predictions,
                selected_store=selected_store,
                selected_item=selected_item,
            )

            # Handle empty results before calculating metrics or plotting.
            if filtered_predictions.empty:
                st.warning(
                    "No predictions match the selected store and item. "
                    "Choose another combination."
                )

            else:
                # Calculate evaluation metrics for this store-item series.
                filtered_metrics = calculate_sales_metrics(
                    filtered_predictions
                )

                # Show the metrics for the selected combination.
                show_metric_cards(filtered_metrics)

                # Prepare the selected series for the time-series chart.
                filtered_chart = prepare_prediction_chart(
                    filtered_predictions
                )

                # Display actual and predicted sales over time.
                st.markdown("**Actual vs Predicted Sales**")

                st.line_chart(
                    filtered_chart.set_index("date")[
                        ["actual_sales", "predicted_sales"]
                    ]
                )

                # Show the matching prediction records in a table.
                with st.expander("View filtered prediction records"):
                    st.dataframe(
                        filtered_predictions.sort_values("date"),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Export only the currently selected store-item predictions.
                st.download_button(
                    label="Download filtered predictions (CSV)",
                    data=make_csv_download(
                        filtered_predictions.sort_values("date")
                    ),
                    file_name=(
                        f"predictions_store_{selected_store}_"
                        f"item_{selected_item}.csv"
                    ),
                    mime="text/csv",
                    help=(
                        "Export actual and predicted sales for the "
                        "selected store and item."
                    ),
                    key="download_filtered_predictions",
                )
                
    
    # -----------------------------------------------------------------------
    # TAB 5: LIVE FORECASTING THROUGH FASTAPI
    # -----------------------------------------------------------------------

    with live_forecast_tab:
        # Introduce the live forecasting feature.
        st.subheader("Generate a Live Demand Forecast")

        # Explain that this tab calls the trained model through FastAPI.
        st.caption(
            "Request a new demand estimate from the trained XGBoost model "
            "using the selected store, item, and forecast date."
        )

        # Check API health and display whether the service is available.
        try:
            health = check_api_health()

            # Display the API's status and model artifact availability.
            if (
                health.get("status") == "ok"
                and health.get("model_artifact_exists") is True
            ):
                st.success("Forecasting API is available and the model artifact exists.")
            else:
                st.warning(
                    "The API responded, but its health status indicates "
                    "that the service or model may be unavailable."
                )

        # Handle an unavailable API without crashing the dashboard.
        except httpx.HTTPError as error:
            st.error(
                "Cannot connect to the forecasting API. "
                "Make sure the FastAPI server is running at "
                "http://127.0.0.1:8000."
            )
            st.caption(f"Technical details: {error}")

        # Add an input form so the user submits all values together.
        with st.form("live_forecast_form"):
            # Display store and item selectors side by side.
            store_column, item_column = st.columns(2)

            # Let the user choose a store from the supported range.
            selected_forecast_store = store_column.selectbox(
                "Store ID",
                options=list(range(1, 11)),
                index=0,
                key="live_forecast_store",
            )

            # Let the user choose an item from the supported range.
            selected_forecast_item = item_column.selectbox(
                "Item ID",
                options=list(range(1, 51)),
                index=0,
                key="live_forecast_item",
            )

            # Default to the next calendar day.
            selected_forecast_date = st.date_input(
                "Forecast date",
                value=date.today(),
                key="live_forecast_date",
                help=(
                    "Choose the date for which you want to estimate demand. "
                    "The model requires at least 30 consecutive historical "
                    "sales observations before this date."
                ),
            )

            # Submit all selected inputs to the forecasting API.
            submitted = st.form_submit_button(
                "Generate Forecast",
                type="primary",
                use_container_width=True,
            )

        # Request a forecast only after the user submits the form.
        if submitted:
            try:
                # Send the selected values through the API client.
                forecast_result = request_forecast(
                    store=selected_forecast_store,
                    item=selected_forecast_item,
                    forecast_date=selected_forecast_date,
                )

                # Display a success message when the request completes.
                st.success("Forecast generated successfully.")

                # Display the returned prediction prominently.
                st.metric(
                    "Predicted demand (units)",
                    f"{forecast_result['predicted_sales']:.2f}",
                    help=(
                        "Estimated demand from XGBoost. "
                        "This is a prediction, not a guarantee of actual sales."
                    ),
                )

                # Display the identifiers and date used for the forecast.
                result_column, item_result_column, date_result_column = (
                    st.columns(3)
                )

                result_column.metric(
                    "Store",
                    str(forecast_result["store"]),
                )

                item_result_column.metric(
                    "Item",
                    str(forecast_result["item"]),
                )

                date_result_column.metric(
                    "Forecast date",
                    str(forecast_result["date"]),
                )

                # Identify the model that produced the result.
                st.caption(
                    f"Forecasting model: {forecast_result['model_name']}"
                )

                # Let the user download the forecast response as JSON.
                st.download_button(
                    label="Download forecast result (JSON)",
                    data=pd.Series(forecast_result).to_json(indent=2),
                    file_name=(
                        f"forecast_store_{selected_forecast_store}_"
                        f"item_{selected_forecast_item}_"
                        f"{selected_forecast_date.isoformat()}.json"
                    ),
                    mime="application/json",
                    key="download_live_forecast",
                )

            # Handle HTTP errors such as validation failures and server errors.
            except httpx.HTTPStatusError as error:
                st.error(
                    f"The forecasting API returned HTTP "
                    f"{error.response.status_code}."
                )

                # Show the API's response body when available.
                st.code(error.response.text or "No error details were returned.")

            # Handle connection failures and request timeouts.
            except httpx.RequestError as error:
                st.error(
                    "Could not reach the forecasting API. "
                    "Check that the server is running and try again."
                )
                st.caption(f"Technical details: {error}")

            # Handle unexpected response structures or other request issues.
            except (KeyError, ValueError, TypeError) as error:
                st.error(
                    f"The forecast response could not be processed: {error}"
                )


    # -----------------------------------------------------------------------
    # 6. DATA PREVIEW AND FOOTER
    # -----------------------------------------------------------------------

    # Place a general data preview at the bottom of the dashboard.
    with st.expander("Preview raw XGBoost prediction records"):
        st.dataframe(
            predictions.head(100),
            use_container_width=True,
            hide_index=True,
        )

        # Tell users how many rows are shown versus available.
        st.caption(
            f"Showing the first {min(100, len(predictions)):,} of "
            f"{len(predictions):,} valid prediction records."
        )

    # Add a footer describing the project's forecasting workflow.
    st.divider()

    st.caption(
        "Inventory & Demand Forecasting System | "
        "Python · Pandas · Scikit-learn · XGBoost · Streamlit"
    )


# ---------------------------------------------------------------------------
# 6. APPLICATION ENTRY POINT
# ---------------------------------------------------------------------------

# Execute the dashboard only when Streamlit runs this module as a script.
if __name__ == "__main__":
    main()
