# Inventory & Demand Forecasting System
## Forecast Error Analysis Findings

### 1. Evaluation overview

The XGBoost model was evaluated on **182,500**
predictions.

| Metric | Result |
|---|---:|
| MAE | 6.0858 |
| RMSE | 7.9132 |
| WAPE | 10.3474% |
| Mean error (actual - predicted) | 0.1108 |
| Actual total sales | 10,733,740 |
| Predicted total sales | 10,713,513.26 |
| Predicted minus actual sales | -20,226.74 |

A positive mean error indicates a tendency to underpredict demand on average.
A negative mean error indicates a tendency to overpredict demand on average.

### 2. Forecasting error by time

- **Best weekday by WAPE:** Sunday
  (9.5602%).
- **Worst weekday by WAPE:** Monday
  (11.5901%).
- **Best month by WAPE:** July
  (9.2556%).
- **Worst month by WAPE:** January
  (12.5796%).

These results describe the 2017 test period. They do not establish that
the same patterns will necessarily continue in future years.

### 3. Store and item analysis

- **Highest-WAPE store:** Store 7
  (12.3301%).
- **Highest-WAPE item:** Item 5
  (17.5805%).
- **Highest-WAPE store-item combination:** Store
  7, Item
  5
  (22.3731%).

These groups are candidates for further investigation. High WAPE alone does
not prove that a product experienced stockouts or excess inventory.

### 4. Model comparison

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| XGBoost | 6.0858 | 7.9132 | 10.3474% |
| Linear Regression | 7.0660 | 9.2381 | 12.0139% |
| Seasonal naive (7 days) | 8.8415 | 11.5967 | 15.0326% |
| Previous-day naive | 11.2447 | 15.3892 | 19.1188% |

The best model by WAPE in the saved comparison is **XGBoost**.

### 5. Recommended next investigations

1. Investigate why Item 5 has relatively high WAPE.
2. Examine demand patterns for Store 7.
3. Inspect daily errors and unusually large misses, not only average metrics.
4. Add stock availability, inventory levels, and supplier lead times before
   drawing conclusions about stockouts or replenishment performance.
5. Use a separate validation period for future model tuning, preserving the
   test period for an unbiased final evaluation.

### 6. Limitations

- The evaluation represents the available 2017 test period.
- The workflow evaluates rolling one-day-ahead predictions using historical
  sales as they become available.
- The dataset does not itself provide inventory on hand, supplier lead times,
  or stockout records.
- Forecast accuracy metrics do not directly measure inventory cost,
  service level, or lost sales.

### 7. Generated artifacts

- `model_comparison.csv`
- `xgboost_error_summary.csv`
- `xgboost_error_by_weekday.csv`
- `xgboost_error_by_month.csv`
- `xgboost_error_by_store.csv`
- `xgboost_error_by_item.csv`
- `xgboost_error_by_store_item.csv`
- `figures/`

