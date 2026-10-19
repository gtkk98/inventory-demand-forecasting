# Data Preparation Report

## Objective

Prepare a validated, reproducible dataset for time-series demand forecasting while preserving the original raw data.

## Input Dataset

- File: `data/raw/train.csv`
- Records: 913,000
- Columns: `date`, `store`, `item`, `sales`
- Date range: 2013-01-01 to 2017-12-31

## Data Quality Findings

- Missing values in required columns: 0
- Invalid dates: 0
- Invalid store and product identifiers: 0
- Invalid or non-finite sales values: 0
- Negative sales values: 0
- Fractional sales values: 0
- Duplicate date-store-product keys: 0
- Store-product combinations with missing dates: 0

## Preparation Decisions

1. Preserved the raw dataset without modification.
2. Standardized `date` to a datetime type during preparation.
3. Standardized `store`, `item`, and `sales` to integer types.
4. Sorted records by date, store, and item.
5. Preserved all 913,000 records.
6. Identified 11,967 records above the IQR upper threshold of 130 units for investigation; these records were not automatically removed.
7. Saved the processed dataset separately for reproducibility.

## Output Artifacts

- `data/processed/sales_clean.csv`
- `reports/cleaning_inspection.csv`
- `reports/time_series_completeness.csv`
- `reports/sales_distribution.csv`
- `reports/potential_sales_outliers.csv`
- `reports/processed_data_validation.csv`

## Conclusion

The processed dataset retains the complete daily history for 10 stores and 50 products over 1,826 calendar days. No records were removed during preparation. The dataset is ready to serve as the input to time-series feature engineering, subject to successful processed-data validation and ongoing checks against data leakage.
